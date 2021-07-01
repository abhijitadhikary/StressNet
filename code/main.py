import argparse
import os
from models import StressNet, DynamicStressNet
from datasets import ThermalLoader
from train import train, resume, evaluate
from initialization import split_videos, extract_frames, split_train_test, annotation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['Stressful', 'Stressful', 'Stressful', 'Calm', 'Calm', 'Calm', 'Calm', 
        'Stressful', 'Calm', 'Stressful', 'Stressful', 'Stressful', 'Calm', 'Stressful',
         'Calm', 'Stressful', 'Calm', 'Stressful', 'Calm', 'Calm']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='training id')
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--truth', type=int, default=6, help='num of select labels')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # initialization
    if not os.path.exists('./video_segments'): 
        videos_list = os.listdir('./videos')
        for video in videos_list: 
            split_videos(video, LABELS, manual=args.truth)
    if not os.path.exists('./dataset'): 
        extract_frames('./video_segments')
    if not os.path.exists('./dataset/test'): 
        split_train_test('./dataset/train')
        annotation('./dataset/train')
        annotation('./dataset/test')

    # dataloaders
    trainloader = DataLoader(ThermalLoader(args, split='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(ThermalLoader(args, split='test'),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloaders = (trainloader, testloader)
    visualizeloader = DataLoader(ThermalLoader(args, split='test'),
        batch_size=1, shuffle=False, num_workers=0)
    # network
    model = DynamicStressNet().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1: # test mode
        testing_accuracy = evaluate(args, model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    else: # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders)
        print('training finished')