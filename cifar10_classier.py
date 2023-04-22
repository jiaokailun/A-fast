import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from net import Net_s,Net_m,Net_l
from resnet import ResNet50, ResNet18, ResNet34
import resnet
from vgg import VGG


net = VGG('VGG16').cuda()
net = nn.DataParallel(net)
state_dict = torch.load(
    'pretrained/vgg16_cifar10.pth')
net.load_state_dict(state_dict)


def load_data_fashion_mnist(batch_size, root='./data'):
    """Download the fashion mnist dataset and then load into memory."""

    #normalize = transforms.Normalize(mean=[0.28], std=[0.35])
    #normalize = transforms.Normalize((0.5,),(0.5,))
    train_augs = transforms.Compose([
        #transforms.RandomCrop(28, padding=2), # 
        #transforms.RandomHorizontalFlip(), #随机图片水平翻转
        transforms.ToTensor(),
        #normalize,
    ])

    test_augs = transforms.Compose([
        transforms.ToTensor(),
        #normalize,
    ])

    #mnist_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_augs)

    mnist_train = torchvision.datasets.ImageFolder(root = 'adv_data_target/adv_data_T_vgg16_cifar10_PGD', transform=train_augs)
    #mnist_test = torchvision.datasets.ImageFolder(root = 'adv_data_target_test', transform=train_augs)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


print('train...')
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../data')


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    net.eval()
    acc_sum, n = 0.0, 0
    total = 0.0
    correct_netT = 0.0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
            

            outputs = net(X.cuda())
            _, predicted = torch.max(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)
            total += (y.cuda()).size(0)
            correct_netT += (predicted == y.cuda()).sum()
        print('Accuracy of the network on TARGET_NET: %.2f %%' %
            (100. * correct_netT.float() / total))
    net.train()
    return acc_sum / n


def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        if epoch > 0 and epoch % lr_period == 0:
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model_fmnist/cifar10_VGG16_defend_PGD.pth')
            # utils.save_model({
            #    'arch': args.model,
            #    'state_dict': net.state_dict()
            # }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))


lr, num_epochs, lr_period, lr_decay = 0.01, 50, 5, 0.1
#optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay)

print('Load the optimal model')
net.load_state_dict(torch.load('model_fmnist/best.pth'))
net = net.to(device)

print('inference test set')
net.eval()
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('Generate test set evaluation files')
with open('result.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
