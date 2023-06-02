import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutil
import cv2
import pytorch_msssim
import torchvision as tv

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """
    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).to(device)
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).to(device)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
     """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    """
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out

def Int_Loss(fused_image, vis_image, inf_image, w1_vis):

    loss_Int: object = F.l1_loss(fused_image,inf_image) + w1_vis * F.l1_loss(fused_image, vis_image)
    # loss_Int: object = F.l1_loss(fused_image, torch.max(inf_image, vis_image))
    return loss_Int


def gradinet_Loss(fused_image, vis_image, inf_image):
    # w2_ev = (w2_ir + w2_vis) /2
    # gradinet_loss = F.l1_loss(w2_ev * gradient(fused_image), torch.max(w2_ir * gradient(inf_image), w2_vis * gradient(vis_image)))
    gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_image)))

    return gradinet_loss

def SSIM_Loss(fuse_image, vis_image, inf_image,w2):
    ssim_loss = pytorch_msssim.msssim
    gradient_vis = gradient(vis_image)
    gradient_ir = gradient(inf_image)
    weight_A = torch.mean(gradient_vis) / (torch.mean(gradient_vis) + torch.mean(gradient_ir))
    weight_B = torch.mean(gradient_ir) / (torch.mean(gradient_vis) + torch.mean(gradient_ir))
    loss_out = weight_A * ssim_loss(vis_image, fuse_image) + weight_B * ssim_loss(inf_image, fuse_image)
    return loss_out

def draw_features(width, height, x):
    # fig = plt.figure(figsize=(16, 16))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(128):
        # plt.subplot(height, width, i + 1)
        # plt.axis('off')
        img = x[0, 0, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # plt.show()
        # print("{}/{}".format(i, width * height))
    # fig.savefig(savename, dpi=100)
    # fig.clf()
    #     plt.close()

def draw_cnn(feature_map, model):
    feature_map = feature_map.cpu().detach().numpy()
    im = feature_map[0, :, :, :]
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    # plt.imshow(im[:, :,0], cmap='viridis')
    plt.imshow(im[:, :, 0], cmap=model)


def disp_feature_image(im, model, x, y, title):
    plt.subplot(2, x, y)
    if model == 'feature':
        draw_cnn(im, 'viridis')
    elif model == 'img':
        draw_cnn(im, 'gray')
    else :
        draw_features(128, 128, im)
    plt.axis('off')
    plt.title(title, fontsize=12, y=-0.12)
    # plt.tight_layout()
    # plt.subplots_adjust(top=1, bottom=0.07, left=0, right=1, hspace=0.12, wspace=0)
    plt.subplots_adjust(top=1, bottom=0.07, left=0, right=1, hspace=0.12, wspace=0.02)
    plt.margins(0, 0)

def disp_image(im, model, x, y):
    plt.figure(figsize=(x, y))
    if model == 'feature':
        draw_cnn(im, 'viridis')
    elif model == 'img':
        draw_cnn(im, 'gray')
    else :
        draw_features(128, 128, im)
    plt.axis('off')
    plt.tight_layout()
    # plt.subplots_adjust(top=1, bottom=0.07, left=0, right=1, hspace=0.12, wspace=0)
    plt.margins(0, 0)
    plt.show()