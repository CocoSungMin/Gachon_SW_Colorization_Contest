import os
import torch.utils.data
import cv2
import tqdm
import time
#from NetWork import *
from dataloader import ColorHintDataset, tensor2im
from unet_resblock import *



class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial'''
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    print('Starting training epoch {}'.format(epoch+1))
    model.train()
    use_cuda = True
    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, data in enumerate(tqdm.tqdm(train_loader)):
        if use_cuda:
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')
            mask = data["mask"].to('cuda')

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint,mask), dim=1)
        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_hint = model(hint_image)
        #loss = 1-criterion(output_hint, gt_image)
        loss = criterion(output_hint, gt_image)
        losses.update(loss.item(), hint_image.size(0))
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 225 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch+1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch+1))


def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()
    use_cuda = True
    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, data in enumerate(tqdm.tqdm(val_loader)):
        if use_cuda:
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')
            mask = data["mask"].to('cuda')

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint,mask), dim=1)
        data_time.update(time.time() - end)
        output_hint = model(hint_image)



        #loss = 1-criterion(output_hint, gt_image)
        loss = criterion(output_hint, gt_image)
        losses.update(loss.item(), hint_image.size(0))

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 100 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))
        out_hint_np = tensor2im(output_hint)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        cv2.imwrite("outputs/outputs/output_"+str(i)+".png", out_hint_bgr)

    print('Finished validation.')
    return losses.avg




def main():

    ## DATALOADER ##
    # Change to your data root directory
    root_path = "data/"
    # Depend on runtime setting
    use_cuda = True


    train_dataset = ColorHintDataset(root_path, 128)
    train_dataset.set_mode("training")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = ColorHintDataset(root_path, 128)
    test_dataset.set_mode("validation")
    test_dataloader = torch.utils.data.DataLoader(test_dataset)



    model = ResAttdU_Net()
    #model = Unet()
    print(model)

    criterion = nn.L1Loss()
    #criterion = pytorch_ssim.SSIM(window_size=11)
    #criterion = pytorch_msssim.MSSSIM()
    optimizer = torch.optim.Adam(model.parameters(), lr=25e-5, weight_decay=0.0)

    # Move model and loss function to GPU
    if use_cuda:
        model.cuda()
        criterion = criterion.cuda()
        model = model.cuda()
    # Make folders and set parameters
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10
    epochs = 150
    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_dataloader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            losses = validate(test_dataloader, model, criterion, save_images, epoch)
        # Save checkpoint and replace old best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-l1-aug-mask-resattdunet-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))



if __name__ == '__main__':
    main()