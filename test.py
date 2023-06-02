import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from models import fusion_model
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from input_data import ImageDataset
import time
from torchvision.utils import save_image
from uitils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
# torch.cuda.set_device(0)


parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="RoadScene_test/ir/", type=str)
parser.add_argument("--visible_dataroot", default="RoadScene_test/vi/", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="./outputs/", type=str)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)
    net = fusion_model.FusionNet().to(device)
    net.load_state_dict(torch.load("./checkpoints/fusion_grad_cross_new_80.pth"))
    net.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    dirname_ir = os.listdir(opt.infrared_dataroot)
    dirname_vi = os.listdir(opt.visible_dataroot)
    tmp_len = len(dirname_ir)
    # if tmp_len >= 50 :
    #     tmp_len = 50
    with torch.no_grad():
        t = []
        for i in range(tmp_len):
            index = i
            if i != 0:
                start = time.time()
            infrared  = Image.open(os.path.join(opt.infrared_dataroot, dirname_ir[i])).convert('L')
            infrared = transform(infrared).unsqueeze(0).to(device)
          
            visible = Image.open(os.path.join(opt.visible_dataroot, dirname_vi[i]))
            visible = transform(visible)
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(visible)
            vis_y_image = vis_y_image.unsqueeze(0).to(device)
            vis_cb_image = vis_cb_image.to(device)
            vis_cr_image = vis_cr_image.to(device)   # show color
        

            fused_img, _, _, _, _ = net(infrared,vis_y_image)
            if i != 0:
                end = time.time()
                print('consume time:', end - start)
                t.append(end - start)
    
            x = torch.squeeze(fused_img, 1)
            fused_img = YCrCb2RGB(x, vis_cb_image, vis_cr_image)  # show color
            
            fused_img = transforms.ToPILImage()(fused_img)
            fused_img.save(os.path.join(opt.output_root, str(dirname_ir[i])))  # show color
        
        print("mean:%s, std: %s" % (np.mean(t), np.std(t)))
