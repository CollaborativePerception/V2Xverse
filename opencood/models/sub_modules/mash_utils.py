import torch
import torch.nn as nn


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
        shouldUseReLU=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if shouldUseReLU:
            if is_batchnorm:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=False))
            else:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=False))
        else:
            if is_batchnorm:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)


    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size, indices=False):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

        self.indices = indices
        if indices:
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
        else:
            self.maxpool_without_argmax = nn.MaxPool2d(2, 2, return_indices=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()

        if self.indices:
            outputs, indices = self.maxpool_with_argmax(outputs)
            return outputs, indices, unpooled_shape
        else:
            outputs = self.maxpool_without_argmax(outputs)
            return outputs

class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size, shouldUseReLU=True):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.up = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1, shouldUseReLU=shouldUseReLU)

    def forward(self, inputs, indices=None, output_shape=None):
        if indices is not None:
            outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        else:
            outputs = self.up(inputs)

        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class QueryEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QueryEncoder, self).__init__()

        self.model = nn.Sequential(
            conv2DBatchNormRelu(in_ch, 512, 1, 1, 0),
            conv2DBatchNormRelu(512, 512, 1, 1, 0),
            conv2DBatchNormRelu(512, out_ch, 1, 1, 0),
        )
    def forward(self, x):
        return self.model(x)

class KeyEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(KeyEncoder,self).__init__()
        self.model = nn.Sequential(
            conv2DBatchNormRelu(in_ch, 512, 1, 1, 0),
            # conv2DBatchNormRelu(512, 512, 1, 1, 0),
            conv2DBatchNormRelu(512, out_ch, 1, 1, 0),
        )
    def forward(self, x):
        return self.model(x)

class SmoothingNetwork(nn.Module):
    def __init__(self, in_ch=32*32+1):
        super(SmoothingNetwork, self).__init__()
        out_ch = in_ch
        self.d32to16 = segnetDown3(in_ch,256,indices=True)
        self.d16to08 = segnetDown3(256,128,indices=True)
        self.d08to16 = segnetUp3(128,256)
        self.d16to32 = segnetUp3(256,out_ch)

    def forward(self, distAB):
        Da16d,Da_i16,Da_s16 = self.d32to16(torch.nn.Softmax(1)(distAB))
        Da08d,Da_i08,Da_s08 = self.d16to08(Da16d)
        Da08 = Da08d
        Da16 = self.d08to16(Da08,Da_i08,Da_s08)
        Da32 = self.d16to32(Da16,Da_i16,Da_s16)
        return Da32

