# -*- coding: utf-8 -*-
import torch as pt

class DoubleConv(pt.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),  
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),
            )

        self.residual_upsampler = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            pt.nn.InstanceNorm3d(out_ch))

        self.relu=pt.nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.conv(input)+self.residual_upsampler(input))

class Deconv3D_Block(pt.nn.Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = pt.nn.Sequential(
                        pt.nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel,kernel,kernel), 
                                    stride=(stride,stride,stride), padding=(padding, padding, padding), output_padding=0, bias=True),
                        pt.nn.LeakyReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

class SubPixel_Block(pt.nn.Module):
    def __init__(self, upscale_factor=2):
        super(SubPixel_Block,self).__init__()

        self.subpixel=pt.nn.Sequential(
            PixelShuffle3d(upscale_factor),
            pt.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.subpixel(x)

class PixelShuffle3d(pt.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PFSeg3D(pt.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(PFSeg3D, self).__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = pt.nn.MaxPool3d((2,2,2))
        self.conv2 = DoubleConv(32, 32)
        self.pool2 = pt.nn.MaxPool3d((2,2,2))
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = pt.nn.MaxPool3d((2,2,2))
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = pt.nn.MaxPool3d((2,2,2))
        self.conv5 = DoubleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.up10_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_seg = DoubleConv(32, 16)
        self.conv11_seg = pt.nn.Conv3d(16, out_channels, 1)

        self.up6_sr = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_sr = DoubleConv(256, 128)
        self.up7_sr = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_sr = DoubleConv(128, 64)
        self.up8_sr = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_sr = DoubleConv(64, 32)
        self.up9_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_sr = DoubleConv(64, 32)
        self.up10_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_sr = DoubleConv(32, 16)
        self.conv11_sr = pt.nn.Conv3d(16, out_channels, 1)

        # SGM
        self.high_freq_extract=pt.nn.Sequential(
            DoubleConv(in_channels,16),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(16,32),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(32,64),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(64,64),
        )

    def forward(self, x, guidance):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        hfe_seg=self.high_freq_extract(guidance)

        up_6_seg = self.up6_seg(pt.cat([c5,hfe_seg], dim=1))
        merge6_seg = pt.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = pt.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = pt.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = pt.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        up_10_seg = self.up10_seg(c9_seg)
        c10_seg = self.conv10_seg(up_10_seg)
        # c11_seg = self.pointwise(c10_seg)
        c11_seg = self.conv11_seg(c10_seg)
        out_seg = pt.nn.Sigmoid()(c11_seg)

        hfe_sr=self.high_freq_extract(guidance)

        up_6_sr = self.up6_sr(pt.cat([c5,hfe_sr], dim=1))
        merge6_sr = pt.cat([up_6_sr, c4], dim=1)
        c6_sr = self.conv6_sr(merge6_sr)
        up_7_sr = self.up7_sr(c6_sr)
        merge7_sr = pt.cat([up_7_sr, c3], dim=1)
        c7_sr = self.conv7_sr(merge7_sr)
        up_8_sr = self.up8_sr(c7_sr)
        merge8_sr = pt.cat([up_8_sr, c2], dim=1)
        c8_sr = self.conv8_sr(merge8_sr)
        up_9_sr = self.up9_sr(c8_sr)
        merge9_sr = pt.cat([up_9_sr, c1], dim=1)
        c9_sr = self.conv9_sr(merge9_sr)
        up_10_sr = self.up10_sr(c9_sr)
        c10_sr = self.conv10_sr(up_10_sr)
        c11_sr = self.conv11_sr(c10_sr)
        out_sr = pt.nn.ReLU()(c11_sr)

        return out_seg, out_sr