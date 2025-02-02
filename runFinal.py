import argparse
import os
import time

from conf import settings
from utils import get_network, get_test_dataloader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
import time
from FLOPs import get_model_complexity_info


parser = argparse.ArgumentParser(description='2-stage classification model on cifar100')
parser.add_argument('--small', default='mobilenetv2', type=str,  help='small model name under models/')
parser.add_argument('--large', default='seresnet50', type=str,  help='large model name under models/')
parser.add_argument('--small_path', '--sp', default='mobilenetv2', type=str,  help='small model path')
parser.add_argument('--large_path', '--lp', default='mobilenetv2', type=str,  help='large model path')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
parser.add_argument('-threshold', type=float, default=0.10)
args = parser.parse_args()

start_time=time.time()

def main():
    snet = get_network(args.small)
    lnet = get_network(args.large)
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    flopsS, paramsS = get_model_complexity_info(snet,(3,32,32),as_strings=True,print_per_layer_stat=False)
    flopsL, paramsL = get_model_complexity_info(lnet,(3,32,32),as_strings=True,print_per_layer_stat=False)
    snet.load_state_dict(torch.load(args.small_path), args.gpu)
    print(snet)
    snet.eval()
    lnet.load_state_dict(torch.load(args.large_path), args.gpu)
    print(lnet)
    lnet.eval()

    correct_1_small = 0.0
    #correct_5_small = 0.0
    correct_1_large = 0.0
    #correct_5_large = 0.0
    threshold = float(args.threshold)
    total_small = 0
    total_large = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beginTime = time.time()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            image = Variable(image).to(device)
            label = Variable(label).to(device)
            output = snet(image)
            score, pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            if(score>threshold*100):
                correct_1_small += correct[:, :1].sum()
                total_small += args.b
            else:
                output = lnet(image)
                _, pred = output.topk(1, 1, largest=True, sorted=True)
                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()
                correct_1_large += correct[:, :1].sum()
                total_large += args.b
    endTime = time.time()
    testTime = endTime - beginTime
    totalFLOPs = (float(flopsS[:4]) * float(total_small)) + (float(flopsL[:4]) * float(total_large))
    acc = float(correct_1_small + correct_1_large) / 10000
    print("Top1 acc: small model: {}/{} big model: {}/{}".format(correct_1_small, total_small, correct_1_large, total_large))
    #print("Top5 acc: small model: {}/{} big model: {}/{}".format(correct_1_small, total_small, correct_1_large, total_large))
    print("Top1 acc for all the system: ", acc)
    print("Total test time is ", testTime)
    print("Total FLOPs for CNNs is ", totalFLOPs)

if __name__=='__main__':
    main()
