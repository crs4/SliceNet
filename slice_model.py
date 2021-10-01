import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools
import time

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]

''' Pad left/right-most to each other instead of zero padding '''
def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)
class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)
def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )
        #############################################################

def xavier_init(m):
	'''Provides Xavier initialization for the network weights and 
	normally distributes batch norm params'''
	classname = m.__class__.__name__
	if (classname.find('Conv2d') != -1) or (classname.find('ConvTranspose2d') != -1):
		nn.init.xavier_normal_(m.weight.data)
		
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool
                
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4
     

class AConv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, st=(2, 1)):
        super(AConv, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=st, padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
        )

    def forward(self, x):
        return self.layers(x)
    
class Slicing(nn.Module):
    def __init__(self, in_c, out_c, st=(2, 1)):
        super(Slicing, self).__init__()
        self.layer = nn.Sequential(
            AConv(in_c, in_c//2, st=st),
            AConv(in_c//2, in_c//4, st=st),
            AConv(in_c//4, out_c, st=st),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]

        #####HorizonNet-style upsampling        
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3) ## plus 2 on W
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False) ####NB interpolating only W
        x = x[..., factor:-factor] ##minus 2 on W

        ##SIMPLEST
        ##x = F.interpolate(x, size=(x.shape[2], out_w), mode='bilinear', align_corners=False)
        
        return x

class MultiSlicing(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        super(MultiSlicing, self).__init__()
        self.cs = c1, c2, c3, c4 
        
        self.out_scale = out_scale
        self.slc_lst = nn.ModuleList([
            Slicing(c1, c1//out_scale), 
            Slicing(c2, c2//out_scale), 
            Slicing(c3, c3//out_scale),
            Slicing(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.slc_lst, conv_list, self.cs)
        ], dim=1)
        return feature


class SliceNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, full_size = False):
        super(SliceNet, self).__init__()
        self.backbone = backbone
        self.ch_scale = 8        
              
        self.lfeats = 1024 ###default max 

        self.full_size = full_size 

        ##self.out_w_size = 512

        ##if(self.full_size):
            ##self.out_w_size = 1024                     

        self.feature_extractor = Resnet(backbone, pretrained=True)
                
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)##NB c1, c2, c3, c4 do not depend by resolution
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)] ###NB depend by resnet layers depth                                    
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.ch_scale

            if(self.full_size):
                c_last *= 2
         
        ##print('c_last',c_last)

        self.slicing_module = MultiSlicing(c1, c2, c3, c4, self.ch_scale)
        
        self.bi_rnn = nn.LSTM(input_size=c_last,
                              hidden_size=(self.lfeats//2),
                              num_layers=2,
                              dropout=0.5,
                              batch_first=False,
                              bidirectional=True)

        self.drop_out = nn.Dropout(0.5)

        if(self.full_size):
                self.decoder = nn.ModuleList([
                    AConv(self.lfeats, self.lfeats//2, st=(1, 1)),
                    AConv(self.lfeats//2, self.lfeats//4, st=(1, 1)),
                    AConv(self.lfeats//4, self.lfeats//8, st=(1, 1)),
                    AConv(self.lfeats//8, self.lfeats//16, st=(1, 1)),
                    AConv(self.lfeats//16, self.lfeats//32, st=(1, 1)),
                    AConv(self.lfeats//32, self.lfeats//64, st=(1, 1)),
                    AConv(self.lfeats//64, self.lfeats//128, st=(1, 1)),
                    AConv(self.lfeats//128, self.lfeats//256, st=(1, 1)),
                    AConv(self.lfeats//256, 1, st=(1, 1)),
                    ])           
        else:
                self.decoder = nn.ModuleList([
                    AConv(self.lfeats, self.lfeats//2, st=(1, 1)),
                    AConv(self.lfeats//2, self.lfeats//4, st=(1, 1)),
                    AConv(self.lfeats//4, self.lfeats//8, st=(1, 1)),
                    AConv(self.lfeats//8, self.lfeats//16, st=(1, 1)),
                    AConv(self.lfeats//16, self.lfeats//32, st=(1, 1)),
                    AConv(self.lfeats//32, self.lfeats//64, st=(1, 1)),
                    AConv(self.lfeats//64, self.lfeats//128, st=(1, 1)),
                    AConv(self.lfeats//128, 1, st=(1, 1)),
                    ])
                       
        ''' Pad left/right-most to each other instead of zero padding '''       
        wrap_lr_pad(self)
              
        ##self.apply(xavier_init)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x)
                                    
        feature = self.slicing_module(conv_list, x.shape[3])
                               
        feature = feature.permute(2, 0, 1)

        output, hidden = self.bi_rnn(feature)
                  
        output = self.drop_out(output)
        output = output.permute(1, 2, 0) ###restore batch first
        output = output.reshape(output.shape[0], output.shape[1], 1, output.shape[2])
                
        for i, conv in enumerate(self.decoder):
            output = F.interpolate(output, scale_factor=(2,1), mode='nearest')
            output = conv(output)

        depth = output.squeeze(1)
              
                                              
        return depth


if __name__ == '__main__':
    print('testing SliceNet')

    device = torch.device('cuda')

    net = SliceNet('resnet50',full_size = True).to(device)
            
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    print('pytorch_total_params', pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('pytorch_trainable_params', pytorch_trainable_params)

    decoder_params = 0

    for name, param in net.named_parameters():
        if (param.requires_grad and ('decoder' in name) ):
            print(name, param.numel())
            decoder_params += param.numel()

    print('equi decoder parameters', decoder_params)

    rnn_params = 0

    for name, param in net.named_parameters():
        if (param.requires_grad and ('rnn' in name) ):
            print(name, param.numel())
            rnn_params += param.numel()

    print('rnn decoder parameters', rnn_params)

    h_encoder_params = 0

    for name, param in net.named_parameters():
        if (param.requires_grad and ('reduce_height_module' in name) ):
            print(name, param.numel())
            h_encoder_params += param.numel()

    print('height ecoder parameters', h_encoder_params)

    encoder_params = 0

    for name, param in net.named_parameters():
        if (param.requires_grad and ('feature_extractor' in name) ):
            print(name, param.numel())
            encoder_params += param.numel()

    print('resnet encoder parameters', encoder_params)

    ##batch = torch.ones(1, 3, 256, 512).to(device)
    batch = torch.ones(1, 3, 512, 1024).to(device)

    ##with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    out_depth = net(batch)
    torch.cuda.synchronize()
    elapsed_fp = time.time()-t0

    print('time cost',elapsed_fp)
               
    print('out_depth shape', out_depth.shape)

    print('test done')
