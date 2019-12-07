import argparse
import os
import time

import numpy as np
from conf import settings
from utils import get_network, get_training_dataloader, get_train_classified
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


def main():
    net = get_network(args.net, use_gpu=args.gpu)

    net.load_state_dict(torch.load(args.weight), args.gpu)
    #print(net)
    net.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    err_indexes = [[] for i in range(100)]
    err_count =[0 for i in range(100)]
    ind = 0
    training_data = get_train_classified(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
    
    with torch.no_grad():
        for label, image in enumerate(training_data):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            n_image = len(image)
            assert n_image==500
            image = Variable(torch.from_numpy(np.array(image))).to(device)
            label = Variable(torch.from_numpy(np.array([label for i in range(n_image)]))).to(device)
            output = net(image)           
            _ , pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred[:,0].eq(label)
            print(correct)
            #if(not correct[0][0]):
            #    err_indexes[label[0][0]].append(ind)
            #    err_count[label[0][0]] += 1
            #ind += 1
    #print(err_count)
    
    #np.savetxt(args.net+"_errimgs.txt", err_indexes, fmt="%d", delimiter=",")
    #np.savetxt(args.net+"_errstats.txt", err_count, fmt="%d", delimiter=",")
    



if __name__=='__main__':
    main()

