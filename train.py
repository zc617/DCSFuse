import argparse
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import os
from models import fusion_model
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from input_data import ImageDataset
from uitils import *
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins import projector
import matplotlib.pyplot as plt





def hook_func(module, input):
    x = input[0][0]
    x = x.unsqueeze(1)
    global i
    image_batch = tv.utils.make_grid(x, padding=4).cpu()
    image_batch = image_batch.numpy().transpose(1, 2, 0)
    writer.add_image("feature_map:", image_batch, i, dataformats='HWC')
    i += 1

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="./data/TNO_Train_2/ir/", type=str)
parser.add_argument("--visible_dataroot", default="./data/TNO_Train_2/vi/", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default= 80)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
parser.add_argument('--loss_weight', default='[1, 10]', type=str,metavar='N', help='loss weight')

if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    writer = SummaryWriter('./runs/logdir')

    net = fusion_model.FusionNet().to(device)
    # net = nn.DataParallel(net)
    # net =net.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    train_datasets = ImageDataset(opt.infrared_dataroot, opt.visible_dataroot, opt.image_size)
    lens = len(train_datasets)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size,shuffle=True)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    global_step = 0
    w1_vis = 1
    i = 0
    for epoch in range(opt.epoch):
        if epoch % 10 == 1:
           opt.lr=0.1*opt.lr
        net.train()
        # train_tqdm = tqdm(dataloader, total=len(dataloader))
        for index, data in enumerate(dataloader):

            nc, c, h, w = data[0].size()
            infrared = data[0].to(device)
            visible = data[1].to(device)
            fused_img, corr_ir, corr_vis, disp_ir_feature, disp_vis_feature = net(infrared, visible)
            fused_img = clamp(fused_img)
            int_loss = Int_Loss(fused_img, visible, infrared, w1_vis).to(device)
            gradient_loss = gradinet_Loss(fused_img, visible, infrared).to(device)
            t1, t2, = eval(opt.loss_weight)
            loss = t1 * int_loss + t2 * gradient_loss
            runloss += loss.item()
            if epoch == 0 and index == 0:
                writer.add_graph(net, (infrared, visible))
            # global_step += 1
            if index % 200 == 0:  #
                writer.add_scalar('training loss', runloss / 200, epoch * len(dataloader) + index)
                runloss = 0.

            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch % 1 == 0:
            print('write_data, epoch=', epoch)
            print(
                'epoch [{}/{}], images [{}/{}], Int loss is {:.5}, gradient loss is {:.5}, total loss is  {:.5}, lr: {}'.
                format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, int_loss.item(),
                       gradient_loss.item(), loss.item(), opt.lr))
            writer.add_images('IR_images', infrared, dataformats='NCHW')
            writer.add_images('VIS_images', visible, dataformats='NCHW')
            writer.add_images('Fusion_images', fused_img, dataformats='NCHW')
       

    writer.close()
    torch.save(net.state_dict(), './checkpoints/fusion_grad_cross_new_'+str(epoch+1)+'.pth'.format(opt.lr, log_file[2:]))
    print('training is complete!')