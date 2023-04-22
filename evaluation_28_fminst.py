
from __future__ import print_function
import argparse
import os
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack
import foolbox
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import joblib
import torchvision
import torch.utils.data
import torch.utils.data.sampler as sp
from torch.autograd import Variable
import argparse
import os
import numpy as np
import math
from net import Net_s, Net_m, Net_l
from gen_fmnist import  Generator





SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--adv', type=str, help='attack method')
parser.add_argument('--mode', type=str, help='use which model to generate\
    examples. "imitation_large": the large imitation network.\
    "imitation_medium": the medium imitation network. "imitation_small" the\
    small imitation network. ')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \
         --cuda")





testset = torchvision.datasets.FashionMNIST(root='data', train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.28], std=[0.35]),
                                        
                                     ]))

data_list = [i for i in range(1000, 10000)]
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)


device = torch.device("cuda:0" if opt.cuda else "cpu")








C,H,W = 1,32,32

generator =  Generator().cuda()
state_dict = torch.load(
        'save_model_CGAN/FMNIST_10000_best.pth')

generator.load_state_dict(state_dict)

generator = nn.DataParallel(generator)
generator.eval()





def test_adver(net, tar_net, attack, target):

    net.eval()
    tar_net.eval()
    # BIM
    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=100, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # PGD
    elif attack == 'PGD':
        if opt.target:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=4.8,
                nb_iter=100, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
        else:
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                #eps=0.25,  #azure
                eps=4.8,    #minst
                nb_iter=60, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
    # FGSM
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            #eps=0.26, #azure
            eps=4.8,   #minst
            targeted=opt.target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net,
            num_classes=100,
            learning_rate=0.45,
            # loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            binary_search_steps=10,
            max_iterations=12,
            targeted=opt.target)

 

    # ----------------------------------
    # Obtain the accuracy of the model
    # ----------------------------------

    # with torch.no_grad():
    #     correct_netD = 0.0
    #     total = 0.0
    #     net.eval()
    #     for data in testloader:
    #         inputs, labels = data
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #         outputs = net(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct_netD += (predicted == labels).sum()
    #     print('Accuracy of the network on netD: %.2f %%' %
    #             (100. * correct_netD.float() / total))

    # ----------------------------------
    # Obtain the attack success rate of the model
    # ----------------------------------

    correct = 0.0
    total = 0.0
    tar_net.eval()
    total_L2_distance = 0.0


    for i in range(100):
        print("epoch",i)
        N_Class = 10
        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, 100,1,1))).cuda())
        #fixed labels
        y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
        y_fixed = torch.zeros(N_Class**2, N_Class)
        y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).view(N_Class**2, N_Class,1,1).cuda())
        gen_imgs = generator(noise, y_fixed).view(-1,C,28,28)
        labels =  torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9], device='cuda:0')
        outputs = net(gen_imgs)
        _,predicted = torch.max(outputs.data,1)
        if target:
            for j in range(100):
                if predicted[j] == labels[j]:
                    label_rand = torch.randint(0,9,(1,)).to(device)
                    pre =torch.tensor([predicted[j]]).cuda()
                    if pre != label_rand:
                        gen = gen_imgs[j].expand(1,-1,-1,-1).cuda()
                        adv_inputs_ori = adversary.perturb(gen,label_rand)
                        L2_distance = (torch.norm(adv_inputs_ori - gen)).item()
                        total_L2_distance += L2_distance
                        with torch.no_grad():
                            outputs = tar_net(adv_inputs_ori)
                            _, predict = torch.max(outputs.data, 1)
                            total += label_rand.size(0)
                            correct += (predict == label_rand).sum()
        else:           
            for j in range(100):
                
                #pre =torch.tensor([predicted[j]]).cuda()
                #print("j",j)
                if labels[j] == predicted[j]:
                    gen = gen_imgs[j].expand(1,-1,-1,-1).cuda()
                    label = torch.tensor([labels[j]]).cuda()
                    adv_inputs_ori = adversary.perturb(gen, label)
                    L2_distance = (torch.norm(adv_inputs_ori - gen)).item()
                    total_L2_distance += L2_distance
                    with torch.no_grad():
                        outputs = tar_net(adv_inputs_ori)
                        _, predict = torch.max(outputs.data, 1)
                        total += label.size(0)
                        correct += (predict == label).sum()




        if target:
            print('Attack success rate: %.2f %%' %
                (100. * correct.float() / total))
        else:
            print('Attack success rate: %.2f %%' %
                (100.0 - 100. * correct.float() / total))
        print('l2 distance:  %.4f ' % (total_L2_distance / total))





target_net = Net_s().cuda()
target_net = nn.DataParallel(target_net)
state_dict = torch.load(
    'model_fmnist/fmnist_T_nets_defend_T_netm_nets.pth')
target_net.load_state_dict(state_dict)
target_net.eval()




attack_net = Net_s().cuda()
attack_net = nn.DataParallel(attack_net)
state_dict = torch.load(
    'saved_model/T_net_s_fmnist_net_s.pth')
attack_net.load_state_dict(state_dict)


test_adver(attack_net, target_net, opt.adv, opt.target)
