import sys, os
import argparse
import torch
from tqdm import tqdm
from functools import partial

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvmodels
# ref: https://vscode.dev/github/huyvnphan/PyTorch_CIFAR10
sys.path.append('..')
from cifar10_pretrained.cifar10_models import resnet, densenet, mobilenetv2
from dataset.cifar import train_val_transforms, cifar10_config, cifar100_config
from dataset.tiny_imagenet import tiny_imagenet_config, TinyImageNet

os.environ['TORCH_HOME'] = os.path.join('..','pretrained')


def get_cifar10_config(args):
    args.input_dim = [3,32,32] # input_dim[1:] can be arbitrary
    return args

def get_cifar100_config(args):
    args.input_dim = [3,32,32] # input_dim[1:] can be arbitrary
    return args

def get_timagenet_config(args):
    args.input_dim = [3,64,64] # input_dim[1:] can be arbitrary
    return args

DATA_CONFIG_DICT = {
    'cifar10': cifar10_config,
    'cifar100': cifar100_config,
    'timagenet200': tiny_imagenet_config,
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

SWAV_MODEL_DICT = {
    'resnet50': partial(torch.hub.load, 'facebookresearch/swav:main'),
    'resnet50w2': partial(torch.hub.load, 'facebookresearch/swav:main'),
    'resnet50w5': partial(torch.hub.load, 'facebookresearch/swav:main'),
}

IMAGENET_MODEL_DICT = {
    'resnet18': (tvmodels.resnet18, 'IMAGENET1K_V1'),
    'resnet152': (tvmodels.resnet152, 'IMAGENET1K_V2'),
    'wide_resnet101_2': (tvmodels.wide_resnet101_2, 'IMAGENET1K_V2'),
    'regnet_y_128gf': (tvmodels.regnet_y_128gf, 'IMAGENET1K_SWAG_E2E_V1'),
}


def _teacher_loader(args):
    '''
    Train loader for inference/feature extraction on GPU
    '''
    _, transform = train_val_transforms(DATA_CONFIG_DICT[args.dataset]) # validation transforms
    if args.input_dim[0]==1:
        transform = transforms.Compose([
            transform,
            transforms.Lambda(lambda x: torch.cat([x] * 3, dim=0)),
        ])
    
    if args.dataset=='cifar10':
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    elif args.dataset=='cifar100':
        train_set = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    elif args.dataset=='timagenet200':
        data_root = os.path.join('..','data','tiny-imagenet-200')
        train_set = TinyImageNet(data_root, split='train', transform=transform)
        test_set = TinyImageNet(data_root, split='val', transform=transform)
    else:
        raise Exception(f'args.dataset={args.dataset} not found.')
    
    loader_kwargs = {
        'num_workers': 10,
        'pin_memory': True,
    }
    train_loader = DataLoader(train_set, batch_size=min(len(train_set), args.batch_size), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=min(len(test_set), args.batch_size), shuffle=False, **loader_kwargs)
    return train_loader, test_loader


@torch.no_grad()
def teach(args, model, loaders):
    model.eval()
    
    feature_list = []
    for loader in loaders:
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(args.device)
            logits = model(inputs)
            feature_list.append(logits.to('cpu'))
    features = torch.cat(feature_list)
    return features
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Laplacian SSL: Feature Inference with Teacher Models')
    parser.add_argument('--batch_size', type=int, default=10000, 
                        help='batch size')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='experiment dataset: cifar10 / cifar100 / timagenet200')
    parser.add_argument('--source_path', type=str, default='../data', 
                        help='path to data source')
    # teacher
    parser.add_argument('--teacher_arch', type=str, default='densenet161', 
                        help='teacher architecture: resnet50 / densenet161 / resnet50<w2/w5>')
    parser.add_argument('--teacher_pretrain', type=str, default='cifar10', 
                        help='pretraining method: cifar10 (supervised) / imagenet (supervised) / swav (on imagenet)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config_dict = {
        'cifar10': get_cifar10_config,
        'cifar100': get_cifar100_config,
        'timagenet200': get_timagenet_config,
    }
    args = config_dict[args.dataset](args)
    teacher_dim_dict = {
        'cifar10': 10,
        'imagenet': 1000,
        'swav': 1000,
    }
    args.teacher_dim = teacher_dim_dict[args.teacher_pretrain]
    
    # config dir
    args.pretrained_dir = os.path.join('..', 'pretrained', args.dataset)
    if not os.path.exists(args.pretrained_dir): 
        os.makedirs(args.pretrained_dir)
    args.pretrained_name = os.path.join(args.pretrained_dir, f'{args.teacher_arch}_{args.teacher_pretrain}_dim{args.teacher_dim:d}.pt')
        
    # create model
    if args.teacher_pretrain=='swav': # contrastive pretraining
        model = SWAV_MODEL_DICT[args.teacher_arch](args.teacher_arch)
    elif args.teacher_pretrain=='cifar10':
        model = CIFAR10_MODEL_DICT[args.teacher_arch](pretrained=True)
    elif args.teacher_pretrain=='imagenet':
        teacher_model, pretrained_weights = IMAGENET_MODEL_DICT[args.teacher_arch]
        teacher = teacher_model(weights=pretrained_weights)
    else:
        raise Exception(f'args.teacher_pretrain = {args.teacher_pretrain} not found')
    model = model.to(args.device)
    
    # prepare data
    train_loader, test_loader = _teacher_loader(args)
    
    # inference
    features = teach(args, model, [train_loader, test_loader])
    
    # save
    torch.save(features, args.pretrained_name)
    
    print(f'Pretrained features saved to {args.pretrained_name}')
