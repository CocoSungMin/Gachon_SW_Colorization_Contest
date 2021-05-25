import torch
import torch.nn as nn

class upsample_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(upsample_block, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,3,1,1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True ),
            nn.ConvTranspose2d(ch_out , ch_out , 3,2,1,1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True )
        )

    def forward(self, x):
        x = self.up(x)

        return x



class Residual_recurrent_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual_recurrent_block, self).__init__()

        self.RCNN = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out)
            #nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)

        x1 = self.RCNN(x)
        return x + x1



class Attention_block(nn.Module):
    def __init__(self, upsample, downsample, ch_result):
        super(Attention_block, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(upsample, ch_result, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_result)
        )

        self.up = nn.Sequential(
            nn.Conv2d(downsample, ch_result, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_result)
        )

        self.concat = nn.Sequential(
            nn.Conv2d(ch_result, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        s1 = self.skip(g)

        u1 = self.up(x)
        attd = self.relu(s1 + u1)
        attd = self.concat(attd)

        return x * attd


class ResAttdU_Net(nn.Module):
    def __init__(self, img_ch=4, output_ch=3):
        super(ResAttdU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.DownSample1 = Residual_recurrent_block(ch_in=img_ch, ch_out=64 )
        self.DownSample1_1 = Residual_recurrent_block(ch_in=64 , ch_out=64)

        self.DownSample2 = Residual_recurrent_block(ch_in=64, ch_out=128)
        self.DownSample2_1 = Residual_recurrent_block(ch_in=128, ch_out=128)

        self.DownSample3 = Residual_recurrent_block(ch_in=128, ch_out=256)
        self.DownSample3_1 = Residual_recurrent_block(ch_in=256, ch_out=256)

        self.DownSample4 = Residual_recurrent_block(ch_in=256, ch_out=512 )
        self.DownSample4_1 = Residual_recurrent_block(ch_in=512, ch_out=512)

        self.DownSample5 = Residual_recurrent_block(ch_in=512, ch_out=1024)
        self.DownSample5_1 = Residual_recurrent_block(ch_in=1024, ch_out=1024)

        self.Up5 = upsample_block(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(upsample=512, downsample=512, ch_result=256)
        self.UpSample5 = Residual_recurrent_block(ch_in=1024, ch_out=512)
        self.UpSample5_1 = Residual_recurrent_block(ch_in=512, ch_out=512)

        self.Up4 = upsample_block(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(upsample=256, downsample=256, ch_result=128)
        self.UpSample4 = Residual_recurrent_block(ch_in=512, ch_out=256 )
        self.UpSample4_1 = Residual_recurrent_block(ch_in=256, ch_out=256)

        self.Up3 = upsample_block(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(upsample=128, downsample=128, ch_result=64)
        self.UpSample3 = Residual_recurrent_block(ch_in=256, ch_out=128 )
        self.UpSample3_1 = Residual_recurrent_block(ch_in=128, ch_out=128)

        self.Up2 = upsample_block(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(upsample=64, downsample=64, ch_result=32)
        self.UpSample2 = Residual_recurrent_block(ch_in=128, ch_out=64)
        self.UpSample2_1 = Residual_recurrent_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # encoding path
        x1 = self.DownSample1(x)
        x1 = self.DownSample1_1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.DownSample2(x2)
        x2 = self.DownSample2_1(x2)

        x3 = self.Maxpool(x2)
        x3 = self.DownSample3(x3)
        x3 = self.DownSample3_1(x3)

        x4 = self.Maxpool(x3)
        x4 = self.DownSample4(x4)
        x4 = self.DownSample4_1(x4)

        x5 = self.Maxpool(x4)
        x5 = self.DownSample5(x5)
        x5 = self.DownSample5_1(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpSample5(d5)
        d5 = self.UpSample5_1(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpSample4(d4)
        d4 = self.UpSample4_1(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpSample3(d3)
        d3 = self.UpSample3_1(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpSample2(d2)
        d2 = self.UpSample2_1(d2)

        d1 = self.Conv_1x1(d2)

        return d1