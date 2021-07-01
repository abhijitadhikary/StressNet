import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.functional
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.BCELoss().to(device)


def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0
    ###
    losses = []
    accs = []
    ###
    # training
    print('Network training starts ...')
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time(); iter_time = time.time()
        ###
        loss_one_iter = []
        ###
        for i, data in enumerate(trainloader):

            imgs = data['img']; labels = data['label']
            imgs, labels = imgs.to(device), labels.to(device)
            predict = model(imgs)
            loss = criterion(predict, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                    time.time()-iter_time, loss.item()))
                iter_time = time.time()
                loss_one_iter.append(loss.item())
        batch_time = time.time() - batch_time
        
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        # evaluation
        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            ###
            accs.append(testing_accuracy)
            ###
            print('testing accuracy: {:.3f}'.format(testing_accuracy))
            print('-------------------------------------------------')

            if testing_accuracy > best_testing_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, r'C:\Users\6\Desktop\8420Assignment_2\code\{}_checkpoint.pth'.format(args.exp_id))
                best_testing_accuracy = testing_accuracy
                print('new best model saved at epoch: {}'.format(epoch))
                print('-------------------------------------------------')
        ###
        losses.append(np.mean(loss_one_iter))
        ###
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))
    
    ###
    plt.figure()
    plt.title("Loss Analysis")
    plt.plot(range(args.epochs), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(args.exp_id+"_loss_analysis.png")
    plt.show()
    plt.figure()
    plt.title("Accuracy Analysis")
    plt.plot(range(args.epochs), accs)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(args.exp_id+"_test_acc_analysis.png")
    plt.show()
    ###
    #save data
    np.save(args.exp_id + "_loss.npy", np.array(losses))
    np.save(args.exp_id + "_acc.npy", np.array(accs))
    


def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0]).to(device); correct_count = torch.tensor([0.0]).to(device)

    for i, data in enumerate(testloader):
        imgs = data['img']; labels = data['label']
        imgs, labels = imgs.to(device), labels.to(device)
        total_count += labels.size(0)

        with torch.no_grad():
            predict = model(imgs)
            predict_L = torch.round(predict)
            correct_count += (predict_L == labels).sum()
    testing_accuracy = correct_count / total_count

    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = r'C:\Users\6\Desktop\8420Assignment_2\code\{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer
