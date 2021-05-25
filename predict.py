import os
import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
from torch import nn

import matplotlib as plt
import cv2
import tqdm
import time
from l1 import *
from nnArch import *
from attentionUnet import *
from dataloader import ColorHintDataset, tensor2im



class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def main():
    root_path = './data'
    #check_point = '/home/yugioh1118/multimedia/colorNet-pytorch/checkpoints/model-l1-unet-epoch-36-losses-0.017.pth' # check point file directory
    check_point = '/home/yugioh1118/multimedia/colorNet-pytorch/checkpoints/model-attd-msssim-lr4-unet-epoch-33-losses-0.052.pth'
    #check_point = '/home/yugioh1118/multimedia/colorNet-pytorch/checks/simple_U_Net_checkpoints/model-ssim-simple-unet-epoch-20-losses-0.068.pth'
    use_cuda = True

    test_dataset = ColorHintDataset(root_path,128)
    test_dataset.set_mode('validation')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    #model = UNet()
    #model = UnetGenerator(in_dim = 3 , out_dim = 3 , num_filter = 64).cuda()
    #model = AttU_Net().cuda()
    model.load_state_dict(torch.load(check_point))
    os.makedirs('outputs/predict', exist_ok=True)
    os.makedirs('outputs/predict/gt',exist_ok=True)
    os.makedirs('outputs/predict/hint',exist_ok=True)
    os.makedirs('outputs/predict/pred',exist_ok=True)
    print(len(test_dataloader))
    total_psnr = 0.0
    for i , data in enumerate(tqdm.tqdm(test_dataloader)):
        if use_cuda :
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')
        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint), dim=1)

        gt_np = tensor2im(gt_image)
        gt_noise = tensor2im(hint_image)
        gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
        gt_noise = cv2.cvtColor(gt_noise,cv2.COLOR_LAB2BGR)

        output_hint = model(hint_image)
        out_hint_np = tensor2im(output_hint)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        """gt_bgr = cv2.resize(gt_bgr,dsize=(256,256))
        hint_bgr = cv2.resize(out_hint_bgr , dsize=(256,256))
        gt_noise = cv2.resize(gt_noise , dsize=(256,256))"""

        #total_psnr =  total_psnr +PSNR()(gt_image , output_hint)
        cv2.imwrite('outputs/predict/gt/gt_' + str(i) + '.jpg', gt_bgr)
        cv2.imwrite('outputs/predict/pred/pred_' + str(i) + '.jpg', out_hint_bgr)
        cv2.imwrite('outputs/predict/hint/hint_'+str(i)+'.jpg',gt_noise)

    #print(total_psnr/500)

if __name__ == '__main__':
    main()