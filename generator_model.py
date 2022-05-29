import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
from torchsummary import summary


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dp: bool = True):
        '''
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|

        Args:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        '''
        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers = [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)

    def forward(self, x): return x + self.net(x)


class ResidualGenerator(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 3, apply_dp: bool = True, nb_downsampling=9, nb_resblks=2,scale_channels=2):
        '''
        Residual generator architecture consisting downsampling blocks (nb_downsampling), residual blocks (nb_reblks) and upsampling blocks.
        Downsampling blocks consist of Conv-InstanceNorm-ReLU with a kernel size of 3. The # of channels can be scaled in each down/up sampling stage with scale_channels.
        Upsampling blocks consist of TransposeConv-InstanceNorm-ReLU.
        Args:
            in_channels: number of input chamnels, for RGB 3
            out_channels: number of output channels, should be similar to in_channels
            apply_dp: apply dropout in the residual blocks
            nb_downsampling: number of downsampling stages
            nb_resblks: number of residual  blocks in the bottleneck
            scale_channels: Integer scaling factor to change the amount of channels within each up/downsampling stage
        '''

        super().__init__()

        f = 1
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(nb_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * scale_channels * f, kernel_size=3, stride=2, padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * scale_channels * f), nn.ReLU(True)]
            f *= scale_channels

        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels=out_channels * f, apply_dp=apply_dp)
            self.layers += [res_blk]

        for i in range(nb_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f // scale_channels), 3, 2, padding=1, output_padding=1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f // scale_channels)), nn.ReLU(True)]
            f = f // scale_channels

        conv = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=7, stride=1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_downs=9,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 apply_dp=False
                 ):
        '''
        Construct a Unet generator
        Args:
            input_nc:   the number of channels in input images
            output_nc:  the number of channels in output images
            num_downs:  the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf:        the number of filters in the last conv layer
            norm_layer: normalization layer
        '''
        input_nc = in_channels
        output_nc = out_channels
        super(UNetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8,
                                             ngf * 8,
                                             input_nc=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True
                                             )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8,
                                                 ngf * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=apply_dp
                                                 )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4,
                                             ngf * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer
                                             )
        unet_block = UnetSkipConnectionBlock(ngf * 2,
                                             ngf * 4,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer
                                             )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc,
                                             ngf,
                                             input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer
                                             )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False
                 ):
        '''Construct a Unet submodule with skip connections.
        Args:
            outer_nc:           the number of filters in the outer conv layer
            inner_nc:           the number of filters in the inner conv layer
            input_nc:           the number of channels in input images/features
            submodule:          previously defined submodules
            outermost:          if this module is the outermost module
            innermost :         if this module is the innermost module
            norm_layer:         normalization layer
            use_dropout:        if use dropout layers.
        '''
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias
                             )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1
                                        )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias
                                        )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias
                                        )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)

class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels = 3, out_channels= 64, nb_layers = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        """
        Parameters:
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
        """


        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, nb_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** nb_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)




if __name__ == '__main__':
    rand = torch.rand((5, 3, 512, 512),device='cpu')
    model = UNetGenerator(3, 3).to('cpu')
    summary(model, (3, 512, 512),device='cpu')
