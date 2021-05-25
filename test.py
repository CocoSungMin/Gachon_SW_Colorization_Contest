import os

import torch.utils.data

import cv2
import tqdm

from dataloader import ColorHintDataset, tensor2im

from unet_resblock import *

def main():
    root_path = './data'
    check_point = '/home/yugioh1118/multimedia/colorNet-pytorch/checkpoints/model-l1-aug-mask-resattdunet-epoch-122-losses-0.006.pth'
    use_cuda = True

    test_dataset = ColorHintDataset(root_path,128)
    test_dataset.set_mode('testing')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    model = ResAttdU_Net().cuda()
    #model = Unet().cuda()
    model.load_state_dict(torch.load(check_point))
    os.makedirs('outputs/predict', exist_ok=True)
    os.makedirs('outputs/predict/test',exist_ok=True)

    print(len(test_dataloader))
    model.eval()
    for i , data in enumerate(tqdm.tqdm(test_dataloader)):
        if use_cuda :
            l = data["l"].to('cuda')
            ab = data["hint"].to('cuda')
            filename = data["file_name"]
            mask = data["mask"].to('cuda')
        hint_image = torch.cat((l, ab,mask), dim=1)



        output_hint = model(hint_image)
        out_hint_np = tensor2im(output_hint)
        output_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        fname = str(filename).replace("['", '')
        fname = fname.replace("']", '')

        cv2.imwrite('outputs/predict/test/'+str(fname),output_bgr)



if __name__ == '__main__':
    main()