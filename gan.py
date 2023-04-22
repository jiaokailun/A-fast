import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import joblib
from net import Net_s, Net_m, Net_l
import torch.utils.data
import torch.utils.data.sampler as sp
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

train_ds = torchvision.datasets.MNIST('data',
                                      train = True,
                                      transform = transforms,
                                      download = True)

data_list = [i for i in range(7900, 8000)] # fast validation
dataloader = torch.utils.data.DataLoader(train_ds, batch_size = 10, sampler = sp.SubsetRandomSampler(data_list))


###############
#generator
###############
# linear 1: 100--256
# linear 2: 256--512
# linear 2: 512--28*28
# reshape: 28*28--(1,28,28)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.share_conv = nn.Sequential(

            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,512), #64*256  64*512
            nn.ReLU(),
            nn.Linear(512, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()



            # nn.Linear(512,28*28), # 64*512 ->64*28*28
            # nn.Tanh()
        )

        self.label1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        self.G1 = nn.Sequential(
            nn.Linear(513, 28 * 28),
            nn.Tanh()
        )
        self.label2 = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        self.G2 = nn.Sequential(
            nn.Linear(513, 28 * 28),
            nn.Tanh()
        )
        self.label3 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        self.G3 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label4 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        self.G4 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label5 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float)
        self.G5 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label6 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float)
        self.G6 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label7 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float)
        self.G7 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label8 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float)
        self.G8 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label9 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        self.G9 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )
        self.label10 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float)
        self.G10 = nn.Sequential(
            nn.Linear(513, 28 * 28),  # 64*512 ->64*28*28
            nn.Tanh()
        )

    def forward(self, x):
        img_emb = self.share_conv(x)
        print('img_emg_size', img_emb.size())
        img = img_emb.view(-1, 28, 28, 1)
        print('img_size', img.size())
        return img
        print(img_emb.size(0))
        img = self.main(x)
        self.label1 = self.label1.view(10, 1)
        img_emb1 = torch.cat((img_emb, self.label1), 1)
        img1 = self.G1(img_emb1)
        self.label2 = self.label2.view(10, 1)
        img_emb2 = torch.cat((img_emb, self.label2), 1)
        img2 = self.G2(img_emb2)
        self.label3 = self.label3.view(10, 1)
        img_emb3 = torch.cat((img_emb, self.label3), 1)
        img3 = self.G3(img_emb3)
        self.label4 = self.label4.view(10, 1)
        img_emb4 = torch.cat((img_emb, self.label4), 1)
        img4 = self.G4(img_emb4)
        self.label5 = self.label5.view(10, 1)
        img_emb5 = torch.cat((img_emb, self.label5), 1)
        img5 = self.G5(img_emb5)
        self.label6 = self.label6.view(10, 1)
        img_emb6 = torch.cat((img_emb, self.label6), 1)
        img6 = self.G6(img_emb6)
        self.label7 = self.label7.view(10, 1)
        img_emb7 = torch.cat((img_emb, self.label7), 1)
        img7 = self.G7(img_emb7)
        self.label8 = self.label8.view(10, 1)
        img_emb8 = torch.cat((img_emb, self.label8), 1)
        img8 = self.G8(img_emb8)
        self.label9 = self.label9.view(10, 1)
        img_emb9 = torch.cat((img_emb, self.label9), 1)
        img9 = self.G9(img_emb9)
        self.label10 = self.label10.view(10, 1)
        img_emb10 = torch.cat((img_emb, self.label10), 1)
        img10 = self.G10(img_emb10)
        img1 = img1.view(-1, 28, 28, 1)
        img2 = img2.view(-1, 28, 28, 1)
        img3 = img3.view(-1, 28, 28, 1)
        img4 = img4.view(-1, 28, 28, 1)
        img5 = img5.view(-1, 28, 28, 1)
        img6 = img6.view(-1, 28, 28, 1)
        img7 = img7.view(-1, 28, 28, 1)
        img8 = img8.view(-1, 28, 28, 1)
        img9 = img9.view(-1, 28, 28, 1)
        img10 = img10.view(-1, 28, 28, 1)

        return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10




class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.main(x)
        return x


#########################
#target model
#########################

def cal_azure(model, data):
    data = data.view(data.size(0), 784).cpu().numpy()
    output = model.predict(data)
    output = torch.from_numpy(output).cuda().long()
    return output


############################
# loss
############################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)

d_optim = torch.optim.Adam(dis.parameters(),lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(),lr=0.0001)

loss_fn = torch.nn.BCELoss()



D_loss = []
G_loss = []

for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        print(size)
        random_noise = torch.randn(size, 100, device=device)

        print('MNIST_size', img.size()) # 10 ,1 ,28 ,28
        d_optim.zero_grad()

        real_output = dis(img)

        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))


        gen_img1,  gen_img2, gen_img3, gen_img4, gen_img5, gen_img6,  gen_img7,  gen_img8,  gen_img9,  gen_img10 = gen(random_noise)

        fake_output1 = dis(gen_img1.detach())
        fake_output2 = dis(gen_img2.detach())
        fake_output3 = dis(gen_img3.detach())
        fake_output4 = dis(gen_img4.detach())
        fake_output5 = dis(gen_img5.detach())
        fake_output6 = dis(gen_img6.detach())
        fake_output7 = dis(gen_img7.detach())
        fake_output8 = dis(gen_img8.detach())
        fake_output9 = dis(gen_img9.detach())
        fake_output10 = dis(gen_img10.detach())

        d_fake_loss1 = loss_fn(fake_output1, torch.zeros_like(fake_output1))
        d_fake_loss2 = loss_fn(fake_output2, torch.zeros_like(fake_output2))
        d_fake_loss3 = loss_fn(fake_output3, torch.zeros_like(fake_output3))
        d_fake_loss4 = loss_fn(fake_output4, torch.zeros_like(fake_output4))
        d_fake_loss5 = loss_fn(fake_output5, torch.zeros_like(fake_output5))
        d_fake_loss6 = loss_fn(fake_output6, torch.zeros_like(fake_output6))
        d_fake_loss7 = loss_fn(fake_output7, torch.zeros_like(fake_output7))
        d_fake_loss8 = loss_fn(fake_output8, torch.zeros_like(fake_output8))
        d_fake_loss9 = loss_fn(fake_output9, torch.zeros_like(fake_output9))
        d_fake_loss10 = loss_fn(fake_output10, torch.zeros_like(fake_output10))


        d_fake_loss = (d_fake_loss1 + d_fake_loss2 + d_fake_loss3 + d_fake_loss4 + d_fake_loss5 + d_fake_loss6 + d_fake_loss7
                        + d_fake_loss8 + d_fake_loss9 + d_fake_loss10) / 10



        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optim.step()


        real_output = cal_azure(clf, img)
        gen_img = gen(random_noise)


        target_output = cal_azure(clf, gen_img.detach())

        criterion = nn.CrossEntropyLoss()
        d_target_loss = loss_fn(target_output,torch.ones_like(target_output))
        d_target_loss.backward()

        g_optim.zero_grad()

        fake_output1 = dis(gen_img1.detach())
        fake_output2 = dis(gen_img2.detach())
        fake_output3 = dis(gen_img3.detach())
        fake_output4 = dis(gen_img4.detach())
        fake_output5 = dis(gen_img5.detach())
        fake_output6 = dis(gen_img6.detach())
        fake_output7 = dis(gen_img7.detach())
        fake_output8 = dis(gen_img8.detach())
        fake_output9 = dis(gen_img9.detach())
        fake_output10 = dis(gen_img10.detach())

        d_fake_loss1 = loss_fn(fake_output1, torch.ones_like(fake_output1))
        d_fake_loss2 = loss_fn(fake_output2, torch.ones_like(fake_output2))
        d_fake_loss3 = loss_fn(fake_output3, torch.ones_like(fake_output3))
        d_fake_loss4 = loss_fn(fake_output4, torch.ones_like(fake_output4))
        d_fake_loss5 = loss_fn(fake_output5, torch.ones_like(fake_output5))
        d_fake_loss6 = loss_fn(fake_output6, torch.ones_like(fake_output6))
        d_fake_loss7 = loss_fn(fake_output7, torch.ones_like(fake_output7))
        d_fake_loss8 = loss_fn(fake_output8, torch.ones_like(fake_output8))
        d_fake_loss9 = loss_fn(fake_output9, torch.ones_like(fake_output9))
        d_fake_loss10 = loss_fn(fake_output10, torch.ones_like(fake_output10))


        g_loss = (d_fake_loss1 + d_fake_loss2 + d_fake_loss3 + d_fake_loss4 + d_fake_loss5 + d_fake_loss6 + d_fake_loss7
                 + d_fake_loss8 + d_fake_loss9 + d_fake_loss10) / 10

        g_loss.backward()

        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,torch.ones_like(fake_output))
        g_loss.backward()
        g_loss = g_loss + d_target_loss
        g_optim.step()
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:', epoch)
        #gen_img_plot(gen, test_input)
















