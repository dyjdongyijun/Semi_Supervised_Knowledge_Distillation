import sys, os
import argparse
import torch
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvmodels
# ref: https://vscode.dev/github/huyvnphan/PyTorch_CIFAR10
sys.path.append('..')
from cifar10_pretrained.cifar10_models import resnet, densenet, mobilenetv2

os.environ['TORCH_HOME'] = os.path.join('..','pretrained')


def get_mnist_config(args):
    args.num_class = 10
    args.num_train = 60000
    args.num_test = 10000
    args.input_dim = [1,28,28]
    # for transforms.Normalize
    args.normalize = ([0.1307], [0.3081]) #(mean, std)
    return args


def get_cifar10_config(args):
    args.num_class = 10
    args.num_train = 50000
    args.num_test = 10000
    args.input_dim = [3,32,32]
    # for transforms.Normalize
    args.normalize = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) #(mean, std)
    return args


IMAGENET_MODEL_DICT = {
    'resnet18': (tvmodels.resnet18, {'imagenet': 'IMAGENET1K_V1'}),
    'resnet152': (tvmodels.resnet152, {'imagenet': 'IMAGENET1K_V2'}),
    'wide_resnet101_2': (tvmodels.wide_resnet101_2, {'imagenet': 'IMAGENET1K_V2'}),
    'regnet_y_128gf': (tvmodels.regnet_y_128gf, {'imagenet': 'IMAGENET1K_SWAG_E2E_V1'}),
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


def _teacher_loader(args):
    '''
    Train loader of mnist, for inference/feature extraction on GPU
    '''
    if args.input_dim[0]==1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x] * 3, dim=0)),
        ])
    else:
        transform = transforms.ToTensor()
    
    if args.dataset=='mnist':
        train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    elif args.dataset=='cifar10':
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
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
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    # data: mnist
    parser.add_argument('--dataset', type=str, default='mnist', help='experiment dataset: mnist / cifar10')
    parser.add_argument('--source_path', type=str, default='../data', help='path to data source')
    # teacher
    parser.add_argument('--teacher_arch', type=str, default='resnet152', help='teacher architecture for feature extraction: resnet18 / resnet152 / wide_resnet101_2 / regnet_y_128gf')
    parser.add_argument('--teacher_pretrain', type=str, default='imagenet', help='pretrained weights for the teacher architecture: cifar10 / imagenet')
    parser.add_argument('--teacher_dim', type=int, default=1000, help='dimension of pretrained features: if args.teacher_pretrain==imagenet: 1000')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config_dict = {
        'mnist': get_mnist_config,
        'cifar10': get_cifar10_config
    }
    args = config_dict[args.dataset](args)
    
    # config dir
    args.pretrained_dir = os.path.join('..', 'pretrained', args.dataset)
    if not os.path.exists(args.pretrained_dir): 
        os.makedirs(args.pretrained_dir)
    args.pretrained_name = os.path.join(args.pretrained_dir, f'{args.teacher_arch}_{args.teacher_pretrain}{args.teacher_dim:d}.pt')
        
    # create model
    if args.teacher_pretrain=='imagenet':
        teacher, pretrained_weights = IMAGENET_MODEL_DICT[args.teacher_arch]
        model = teacher(weights=pretrained_weights[args.teacher_pretrain])
    elif args.teacher_pretrain=='cifar10':
        model = CIFAR10_MODEL_DICT[args.teacher_arch](pretrained=True)
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
