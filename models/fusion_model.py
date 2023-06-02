import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os





def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size()  
    CORR = []
    Kernel = []
    for i in range(Feature.shape[0]):
        ker = Kernel_tmp[i:i + 1]  
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1) 
        ker = ker.unsqueeze(2).unsqueeze(3) 

        co = F.conv2d(fea, ker.contiguous())  
        CORR.append(co)
        ker = ker.unsqueeze(0)  
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)  
    Kernel = torch.cat(Kernel, 0) 
    return corr, Kernel

corr_size = 16  #

class CorrelationLayer(nn.Module):
    def __init__(self, feat_channel):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(     
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1), 
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, kernel_size=3, padding=1),
        )
        self.Dnorm = nn.InstanceNorm2d(feat_channel)

    def forward(self, F_infrared, F_visible):  #
        # calculate correlation map
        VIS_feat_downsize = self.pool_layer(F_visible)  
        VIS_feat_norm = F.normalize(F_visible)   
        VIS_corr, _ = corr_fun(VIS_feat_downsize, VIS_feat_norm)
        VIS_corr = self.corr_reduce(VIS_corr) 

        IR_feat_downsize = self.pool_layer(F_infrared)
        IR_feat_norm = F.normalize(F_infrared)
        IR_corr, _ = corr_fun(IR_feat_downsize, IR_feat_norm)
        IR_corr = self.corr_reduce(IR_corr)

        return IR_corr, VIS_corr  

class  Feature_extraction(nn.Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()
        self.conv1_1 = nn.Sequential(
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(1, 16, 3, 1, 0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     )
        self.conv1_2 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(16, 16, 3, 1, 0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     )
        self.conv1_3 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(32, 16, 3, 1, 0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     )
        self.conv1_4 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(48, 16, 3, 1, 0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     )  # output 64*64   # vis
        self.conv1_5 = nn.Sequential(   # nn.ReflectionPad2d(1),
                                     nn.Conv2d(64, 64, 1, 1, 0),
                                     # nn.BatchNorm2d(64),
                                     nn.Tanh(),
                                     )  # output 64*64   # vis
    def forward(self,img):
        conv1 = self.conv1_1(img)
        conv2 = self.conv1_2(conv1)
        conv3 = self.conv1_3(torch.cat((conv2, conv1), dim=1))
        conv4 = self.conv1_4(torch.cat((conv3, torch.cat((conv2, conv1), dim=1)), dim=1))
        conv4 = torch.cat((conv4, torch.cat((conv3, torch.cat((conv2, conv1), dim=1)),dim=1)),dim=1)
        conv5 = self.conv1_5(conv4)
        return  conv5

class Feature_reconstruction(nn.Module):
    def __init__(self):
        super(Feature_reconstruction, self).__init__()
        self.conv3_1 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3, 1, 0),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU()
                                     )
        self.conv3_2 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 64, 3, 1, 0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU()
                                     )
        self.conv3_3 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(64, 32, 3, 1, 0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU()
                                     )
        self.conv3_4 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(32, 16, 3, 1, 0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU()
                                     )
        self.conv3_5= nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(16, 1, 3, 1, 0),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU()
                                    )
    def forward(self,feature):
        conv3_1 = self.conv3_1(feature)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        conv3_5 = self.conv3_5(conv3_4)
        return conv3_5

def cross_funtion(corr_ir, corr_vis, ir_feature, vis_feature):
    adp_pool = nn.AdaptiveAvgPool2d(1)

    corr_ir_pool = adp_pool(torch.sigmoid(corr_ir))
    corr_vis_pool = adp_pool(torch.sigmoid(corr_vis))

    corr_vis_pool = 1 + corr_vis_pool / (corr_ir_pool + corr_ir_pool)
    corr_ir_pool = 1 + corr_ir_pool / (corr_ir_pool + corr_ir_pool)

    ir_feature_temp = corr_ir_pool * ir_feature
    vis_feature_temp = corr_vis_pool * vis_feature

    ir_feature = ir_feature_temp
    vis_feature = vis_feature_temp

    return ir_feature, vis_feature

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        channels = [16, 32, 48, 64, 128]
        self.corr_layer4 = CorrelationLayer(feat_channel=channels[3])
        self.feature = Feature_extraction()
        self.reconstruction = Feature_reconstruction()
        self.adp_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, infrared, visible):
        ir_feature = self.feature(infrared)
        vis_feature = self.feature(visible)
        corr_ir, corr_vis = self.corr_layer4(ir_feature, vis_feature)

        disp_ir_feature = ir_feature
        disp_vis_feature = vis_feature
   
        #
        ir_feature, vis_feature = cross_funtion(corr_ir, corr_vis, ir_feature, vis_feature)

        fusion_feature = torch.cat((ir_feature, vis_feature), dim=1)
        fusion_image = self.reconstruction(fusion_feature)
        return fusion_image, corr_ir, corr_vis, disp_ir_feature, disp_vis_feature

