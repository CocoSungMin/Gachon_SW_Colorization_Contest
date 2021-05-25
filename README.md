import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.BatchNorm2d(input_encoder)

        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
            nn.BatchNorm2d(input_decoder)
        )

        self.conv_attn = nn.Sequential(
            nn.Conv2d(output_dim, 1, 1),
            nn.BatchNorm2d(output_dim),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        out = self.sig(out*x2)
        return out

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=128):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters*1),
            nn.ReLU(),
            nn.Conv2d(filters*1, filters*1, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters*1, kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters*1, filters*2, 2, 1)
        self.residual_conv_2 = ResidualConv(filters*2, filters*4, 2, 1)
        self.residual_conv_3 = ResidualConv(filters*4 , filters*8 ,2,1)


        #self.bridge = ResidualConv(filters*3, filters*4, 2, 1)
        self.bridge = ResidualConv(filters * 8, filters * 16, 2, 1)

        self.upsample_1 = Upsample(filters*16, filters*16, 2, 2)
        self.up_residual_conv1 = ResidualConv(filters*16 + filters*8, filters*8, 1, 1)



        self.upsample_2 = Upsample(filters*8, filters*8, 2, 2)
        self.attd_1 = AttentionBlock(filters * 8, filters * 16, filters*8)
        self.up_residual_conv2 = ResidualConv(filters*8 + filters*4, filters*4, 1, 1)

        self.upsample_3 = Upsample(filters*4, filters*4, 2, 2)
        self.up_residual_conv3 = ResidualConv(filters*4 + filters*2, filters*2, 1, 1)

        self.upsample_4 = Upsample(filters*2 , filters*2 , 2, 2)
        self.attd_2 = AttentionBlock(filters * 2,  filters * 4, filters*2)
        self.up_residual_conv4 = ResidualConv(filters*2 + filters*1, filters*1 , 1,1)



        self.output_layer = nn.Sequential(
            nn.Conv2d(filters*1, 3,3 ,1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        # Bridge
        x5 = self.bridge(x4)
        # Decode

        x5 = self.upsample_1(x5)
        x5 = torch.cat([x5, x4], dim=1)
        x6 = self.up_residual_conv1(x5)

        #x7 = self.attd_1(x3 , x7)
        x6 = self.upsample_2(x6)
        a1 = self.attd_1(g = x6 , x = x3)
        x6 = torch.cat([a1, x3], dim=1)
        x7 = self.up_residual_conv2(x6)

        x7 = self.upsample_3(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x8 = self.up_residual_conv3(x7)

        x8 = self.upsample_4(x8)
        a2 = self.attd_2(g = x8 , x = x1)
        x8 = torch.cat([a2 , x1] , dim=1)
        x9 = self.up_residual_conv4(x8)


        output = self.output_layer(x9)

        return output
