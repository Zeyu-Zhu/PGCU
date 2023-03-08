import torch
import torch.nn as nn
import torch.optim as opt

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from utils.visualize import Evaluate
from model.PanNet import PanNet, PanNet_PGCU


# global config
device = 'cuda:0'
epoches = 100
batch_size = 32
evaluater = Evaluate('PanNet', 'WV3', device)
# prepare dataset&dataloader
data_root = '/home/cxy/pen-sharpening/GAU/data/WV3_data'
train_pan = 'train/pan'
train_ms = 'train/ms'
test_pan = 'test/pan'
test_ms = 'test/ms'
train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# PanNet config
p_weight_decay = 1e-5
p_learning_rate = 5e-4
p_lossFun = nn.MSELoss()
PanNet = PanNet(4).to(device)
p_optimizer = opt.Adam(PanNet.parameters(), lr=p_learning_rate, weight_decay=p_weight_decay)
scheduler_1 = torch.optim.lr_scheduler.StepLR(p_optimizer, step_size=100, gamma=0.1)

# PanNet_GAU config
g_weight_decay = 1e-5
g_learning_rate = 5e-4
g_lossFun = nn.MSELoss()
PanNet_PGCU = PanNet_PGCU(4, 128).to(device)
g_optimizer = opt.Adam(PanNet_PGCU.parameters(), lr=g_learning_rate, weight_decay=g_weight_decay)
scheduler_1 = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=100, gamma=0.1)

# record trainning&testing
p_train_loss = []
p_test_loss = []
g_train_loss = []
g_test_loss = []


# trainning
for epoch in tqdm(range(epoches)):
    # trainning
    p_loss = 0
    PanNet.train()
    g_loss = 0
    PanNet_PGCU.train()
    for label, pan, lrms, up_ms, hpan, hlrms in tqdm(train_loader):
        label = torch.Tensor(label).to(device).float()
        pan = torch.Tensor(pan).to(device).float()
        hpan = torch.Tensor(hpan).to(device).float()
        lrms = torch.Tensor(lrms).to(device).float()
        hlrms = torch.Tensor(hlrms).to(device).float()
        # PanNet
        out = PanNet.forward(pan, lrms, hlrms, hpan)
        loss = p_lossFun(out, label)
        p_optimizer.zero_grad()
        loss.backward()
        p_optimizer.step()
        p_loss += loss.item()
        # PanNet_PGCU
        out, up_ms = PanNet_PGCU.forward(pan, lrms, hpan)
        loss_1 = g_lossFun(out, label)
        # optional: for residual structure
        loss_2 =  g_lossFun(up_ms, label)
        loss = loss_1 + loss_2
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()
        g_loss += loss_1.item()
    p_train_loss.append(p_loss/train_loader.__len__())
    g_train_loss.append(g_loss/train_loader.__len__())
    print('epoch:'+str(epoch), 
          'PanNet train loss:'+str(p_loss/train_loader.__len__()), 
          'PanNet_PGCU train loss:'+str(g_loss/train_loader.__len__()))
    # testing
    if epoch%10 == 0:
        p_loss = 0
        PanNet.eval()
        g_loss = 0
        PanNet_PGCU.eval()
        for label, pan, lrms, up_ms, hpan, hlrms in tqdm(test_loader):
            label = torch.Tensor(label).to(device).float()
            pan = torch.Tensor(pan).to(device).float()
            lrms = torch.Tensor(lrms).to(device).float()
            hpan = torch.Tensor(hpan).to(device).float()
            hlrms = torch.Tensor(hlrms).to(device).float()
            # PanNet
            out = PanNet.forward(pan, lrms, hlrms, hpan)
            loss = p_lossFun(out, label)
            p_loss += loss.item()
            # PanNet_GAU
            out, up_ms = PanNet_PGCU.forward(pan, lrms, hpan)
            loss = g_lossFun(out, label)
            g_loss += loss.item()
        p_test_loss.append(p_loss/test_loader.__len__())
        g_test_loss.append(g_loss/test_loader.__len__())
        print('epoch:'+str(epoch), 
              'PanNet test loss:'+str(p_loss/test_loader.__len__()), 
              'PanNet_PGCU test loss:'+str(g_loss/test_loader.__len__()))
    evaluater.visualize(p_train_loss, p_test_loss, g_train_loss, g_test_loss, PanNet, PanNet_PGCU)