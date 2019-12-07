""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#from dataset import CIFAR100Train, CIFAR100Test

def get_network(name, use_gpu=True):
    """ return given network
    """

    if name == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif name == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif name == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif name == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif name == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif name == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif name == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif name == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif name == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif name == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif name == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif name == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif name == 'xception':
        from models.xception import xception
        net = xception()
    elif name == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif name == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif name == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif name == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif name == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif name == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif name == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif name == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif name == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif name == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif name == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif name == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif name == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif name == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif name == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif name == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif name == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif name == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif name == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif name == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif name == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif name == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif name == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif name == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif name == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif name == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_train_classified(mean, std):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    character = [[] for i in range(100)]
    dataset = []
    for (X, Y) in zip(cifar100_training.train_data, cifar100_training.train_labels):  # 将train_set的数据和label读入列表
        dataset.append(list((X, Y)))

    for X, Y in dataset:
        character[Y].append(X)  # 32*32*3
    
    return character

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
