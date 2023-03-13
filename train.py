import argparse
import logging
import math
import os, sys
import random
import shutil
# from functools import partial
import time, datetime
# from collections import OrderedDict
from ml_collections import config_dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

import torchvision.models as tvmodels
# ref: https://vscode.dev/github/huyvnphan/PyTorch_CIFAR10
sys.path.append('..')
from cifar10_pretrained.cifar10_models import resnet, densenet, mobilenetv2

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_seed(args):
    if args.seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
            
            
def initiate_logging(args):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.result_dir, exist_ok=True)
        
    logging.basicConfig(
        filename=os.path.join(args.result_dir, 'log.txt'), 
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
        force=True,
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",
    )
    # logger.info(dict(args._get_kwargs()))
    logger.info(str(args))
    
    wandb.init(
        project="FixMatch_RKD", 
        entity="graph_based_ssl", 
        reinit=True, 
        tags=['test_run'],
    )
    wandb.run.name = args.exp_name
    wandb.config.update(args)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def process_config(args):
    args = config_dict.ConfigDict(vars(args))
    
    # preprocess
    args.label_per_class = args.num_labeled//args.num_classes
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    
    # tags 
    args.tags = config_dict.ConfigDict({
        'dataset': f'{args.dataset}_ncls{args.num_classes}_lpc{args.label_per_class}_{args.labeler}',
        'arch': f'{args.arch}-' + 'x'.join(map(str, args.input_dim)) + '-' + '-'.join(map(str, args.hidden_dim)),
        'dac': f'fixmatch_lambda-u{args.lambda_u:.1e}_pslab-thres{args.threshold:.2f}',
        'rkd': f'rkd-{args.rkd_edge}_{args.rkd_edge_min}_lambda{args.rkd_lambda:.1e}-p{args.rkd_norm:d}',
        'pretrain': f'{args.teacher_arch}_{args.teacher_data}_dim{args.teacher_dim:d}_{args.teacher_mode}',
        'train': f'lr{args.lr:.1e}_epo{args.epochs:d}_bs{args.batch_size:d}_wd{args.wdecay:.1e}',
        'random': f'seed{args.seed:d}'
    })
    
    # exp_name
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H%M")
    args.exp_name = f'{args.tags.dataset}__{args.tags.arch}__{args.tags.dac}__{args.tags.rkd}__{args.tags.pretrain}__{args.tags.train}__{args.tags.random}__{timestamp}'
    args.result_dir = os.path.join(args.out, args.exp_name)
    
    return args


def device_config(args):
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device
    return args


def model_config(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.input_dim = [3,32,32]
        args.num_train = 50000
        args.num_test = 10000
        if args.arch == 'wideresnet':
            args.hidden_dim = [10]
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.hidden_dim = [10]
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.input_dim = [3,32,32]
        args.num_train = 50000
        args.num_test = 10000
        if args.arch == 'wideresnet':
            args.hidden_dim = [100]
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.hidden_dim = [100]
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    
    return args


def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                        depth=args.model_depth,
                                        width=args.model_width,
                                        num_classes=args.num_classes)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6
    ))
    return model


def get_offline_teacher(args):
    pretrained_name = os.path.join(args.pretrain_path, args.dataset, f'{args.teacher_arch}_{args.teacher_data}{args.teacher_dim:d}.pt')
    teacher = torch.load(pretrained_name)[:args.num_train]
    return teacher.float()
    

IMAGENET_MODEL_DICT = {
    'resnet18': (tvmodels.resnet18, 'IMAGENET1K_V1'),
    'resnet152': (tvmodels.resnet152, 'IMAGENET1K_V2'),
    'wide_resnet101_2': (tvmodels.wide_resnet101_2, 'IMAGENET1K_V2'),
    'regnet_y_128gf': (tvmodels.regnet_y_128gf, 'IMAGENET1K_SWAG_E2E_V1'),
}

CIFAR10_MODEL_DICT = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'densenet121': densenet.densenet121,
    'densenet169': densenet.densenet169,
    'densenet161': densenet.densenet161,
    'mobilenet_v2': mobilenetv2.mobilenet_v2,
}

def get_online_teacher(args):
    if args.teacher_data=='imagenet':
        teacher_model, pretrained_weights = IMAGENET_MODEL_DICT[args.teacher_arch]
        teacher = teacher_model(weights=pretrained_weights)
    elif args.teacher_data=='cifar10':
        teacher = CIFAR10_MODEL_DICT[args.teacher_arch](pretrained=True)
    else:
        raise Exception(f'args.teacher_pretrain = {args.teacher_pretrain} not found')
    return teacher


def rkd_loss(args, features_model, features_teacher):
    assert len(features_model)==len(features_teacher)
    num_smp = len(features_teacher)//2
    fea_model1, fea_model2 = features_model.chunk(2)
    fea_model1, fea_model2 = fea_model1[:num_smp], fea_model2[:num_smp]
    fea_teacher1, fea_teacher2 = features_teacher.chunk(2)
    fea_teacher1, fea_teacher2 = fea_teacher1[:num_smp], fea_teacher2[:num_smp]
    
    if args.rkd_edge=='cos':
        _l2norm = lambda x: F.normalize(x, p=2.0, dim=1) #(n/2, dim_fea)
        edge_teacher = (_l2norm(fea_teacher1) * _l2norm(fea_teacher2)).sum(dim=1) #(n/2,)
    elif args.rkd_edge=='maxmatch':
        pslabel = lambda logit: torch.argmax(logit, dim=1)
        edge_teacher = (pslabel(fea_teacher1)==pslabel(fea_teacher2)).float()
    else:
        raise Exception(f'args.rkd_edge = {args.rkd_edge} not found')
    edge_teacher = edge_teacher.detach()
    vmask = edge_teacher > (args.rkd_edge_min - 1.0)
    edge_teacher = edge_teacher[vmask]
    edge_model = (fea_model1[vmask] * fea_model2[vmask]).sum(dim=1)
    
    if any(vmask):
        loss_rkd = torch.norm(edge_teacher - edge_model, p=args.rkd_norm)/vmask.sum()  
    else:
        loss_rkd = torch.Tensor([0.0]).to(args.device)
    return loss_rkd
    

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'], 
                        help='architecture name')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'], 
                        help='dataset name')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--eval_step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument("--expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--labeler', default='class', type=str,
                        choices=['unif', 'class', 'active'], 
                        help='labele selection: unif=uniform over all samples / class=uniform per class / active=active learning')
    parser.add_argument('--lambda_u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size (=mu * batch_size)')
    parser.add_argument('--nesterov', action='store_false', default=True,
                        help='not use nesterov momentum (use by default)')
    parser.add_argument('--no_progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--num_labeled', type=int, default=100,
                        help='number of labeled data, expected to be a constant multiple of the number of classes')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='number of workers')
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--out', default=os.path.join('..','result'),
                        help='directory to output the result')
    parser.add_argument('--pretrain_path', type=str, default=os.path.join('..','pretrained'),
                        help='path to pretrained model/features')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--rkd_edge', default='cos', type=str,
                        choices=['cos', 'maxmatch'], 
                        help='edges of data graph that characterize pair-wise relation between samples: cos(u,v) = (normalized(t(u),2) * normalized(t(v),2)).sum / maxmatch(u,v) = argmax(t(u))==argmax(t(v))')
    parser.add_argument('--rkd_edge_min', default=0.0, type=float,
                        help='sparsification of data graph: w_e = (w_e >= rkd_tol) * w_e [note w_e >= 0]')
    parser.add_argument('--rkd_lambda', default=0.0, type=float,
                        help='coefficient of relational knowledge distillation loss')
    parser.add_argument('--rkd_norm', default=2, type=int,
                        choices=[1,2], help='order of the norm for RKD loss')
    parser.add_argument('--root', type=str, default=os.path.join('..','data'), 
                        help='path to data source')
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed")
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--teacher_arch', type=str, default='densenet161', 
                        choices=['densenet161', 'resnet152', 'wide_resnet101_2'],
                        help='teacher architecture')
    parser.add_argument('--teacher_data', type=str, default='cifar10', 
                        choices=['cifar10', 'imagenet'],
                        help='pretrained source of teacher models')
    parser.add_argument('--teacher_dim', type=int, default=10, 
                        help='dimension of pretrained features')
    parser.add_argument('--teacher_mode', type=str, default='offline', 
                        choices=['offline', 'online'],
                        help='model of feature extraction: offline=load cached features of original data / online=online feature evaluation for augmented data')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--total_steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--use_ema', action='store_false', default=True,
                        help='not use EMA model (use by default)')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    
    args = parser.parse_args()
    args = device_config(args)
    args = model_config(args)
    args = process_config(args)
    set_seed(args)
    initiate_logging(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # args.writer = SummaryWriter(args.out)

    # dataloaders
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, args.root)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    # optimizer & scheduler
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.result_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level
        )

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True
        )
        
    # teacher
    if args.rkd_lambda > 0.0:
        if args.teacher_mode == 'offline':
            teacher = get_offline_teacher(args)
        else: # args.teacher_mode == 'online'
            teacher = get_online_teacher(args).to(args.device)
    else:
        teacher = None

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, teacher)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, teacher):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_rkd = AverageMeter()
        mask_probs = AverageMeter()
        description = ""
        
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
            
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _, idx_u = next(unlabeled_iter) 
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _, idx_u = next(unlabeled_iter)
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            data_time.update(time.time() - end)
            
            # forward
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # supervised loss
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # dac loss
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            
            # rkd loss
            if args.rkd_lambda > 0.0:
                rkd_features_model = logits_u_w # on gpu 
                if args.teacher_mode == 'offline':
                    rkd_features_teacher = (teacher[idx_u]).to(args.device)
                else: # args.teacher_mode == 'online'
                    rkd_inputs = torch.cat([inputs_x, inputs_u_w]).to(args.device)
                    rkd_features_teacher = teacher(rkd_inputs)
                Lkd = rkd_loss(args, rkd_features_model, rkd_features_teacher) # on gpu
            else:
                Lkd = torch.Tensor([0.0]).to(args.device)

            # loss
            loss = Lx + args.lambda_u * Lu + args.rkd_lambda * Lkd

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_rkd.update(Lkd.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            
            description = "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.eval_step,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_probs.avg
            )
            if not args.no_progress:
                p_bar.set_description(description)
                p_bar.update()
                
        logger.info(description+'\n')

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, top5 = test(args, test_loader, test_model, epoch)
            
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': losses.avg,
                'train_loss_x': losses_x.avg,
                'train_loss_u': losses_u.avg,
                'train_loss_rkd': losses_rkd.avg,
                'mask': mask_probs.avg,
                'test_acc_1': test_acc,
                'test_acc_1_best': best_acc,
                'test_acc_5': top5,
                'test_loss': test_loss,
            })

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.result_dir)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        wandb.finish()


@torch.no_grad()
def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    description = ""

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        data_time.update(time.time() - end)
        model.eval()

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.shape[0])
        top1.update(prec1.item(), inputs.shape[0])
        top5.update(prec5.item(), inputs.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()
        
        description = "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
            batch=batch_idx + 1,
            iter=len(test_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        if not args.no_progress:
            test_loader.set_description(description)
    
    logger.info(description+'\n')
    if not args.no_progress:
        test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
