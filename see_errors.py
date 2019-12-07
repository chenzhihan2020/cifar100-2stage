import argparse
import os
import time

from conf import settings
from utils import get_network, get_training_dataloader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *


parser = argparse.ArgumentParser(description='2-stage classification model on cifar100')
parser.add_argument('-net', default='mobilenetv2', type=str,  help='model name under models/')
parser.add_argument('-weight', default='mobilenetv2.pth', type=str,  help='small model path')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
args = parser.parse_args()

from models import mobilenetv2 as small_net
from models import seresnet50 as large_net
start_time=time.time()

def main():
    net = small_net()
    training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=False
    )

    net.load_state_dict(torch.load(args.weight), args.gpu)
    #print(net)
    net.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    err_indexes = [[] for i in range(100)]
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(training_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            image = Variable(image).to(device)
            label = Variable(label).to(device)
            output = net(image)           
            _ , pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred[:,0].eq(label).cpu().numpy()
            print(correct)
            



if __name__=='__main__':
    main()

