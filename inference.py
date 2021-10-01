import os
import argparse
import numpy as np
from PIL import Image
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
import torch

from slice_model import SliceNet
from misc import tools, eval


def_img = 'example/001ad2ad14234e06b2d996d71bb96fc4_color.png'#
def_gt = 'example/001ad2ad14234e06b2d996d71bb96fc4_depth.png'#

def_pth ='ckpt/resnet50_m3d.pth' 
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default = def_pth,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=False, default = def_img)
    parser.add_argument('--gt_depth', required=False, default = def_gt)
    parser.add_argument('--no_cuda', action='store_true', default = False)
    
    args = parser.parse_args()
        
    device = torch.device('cpu' if args.no_cuda else 'cuda')
       
    net = tools.load_trained_model(SliceNet, args.pth).to(device)
    net.eval()

    # Inferencing   
    # 
    img_pil = Image.open(args.img_glob)

    full_W,full_H = img_pil.size

    H, W = 512,1024

    img_pil = img_pil.resize((W,H), Image.BICUBIC)
    img = np.array(img_pil, np.float32)[..., :3] / 255.
        
   
    with torch.no_grad():
        ####predict depth
        x_img = torch.FloatTensor(img.transpose([2, 0, 1]).copy())       
        x = x_img.unsqueeze(0)
        depth = net(x.to(device))  

        ####create valid mask for Matterport sensor
        depth_gt = np.array(Image.open(args.gt_depth), np.float32)        
        xrange = lambda x: np.linspace(0, 1, x)
        f = interp2d(xrange(full_W), xrange(full_H), depth_gt, kind="linear")##kind="cubic")
        depth_gt = f(xrange(W), xrange(H))        
        depth_gt /= 4000.0 #####matterport scale to meters
        depth_mask = ((depth_gt <= 127.0) & (depth_gt > 0.)).astype(np.uint8)
        depth_mask = torch.FloatTensor(depth_mask.copy())

        
        depth_c = (depth_mask*depth.cpu()).numpy().astype(np.float32).squeeze(0) 
                              
          
        plt.figure(0)
        plt.title('prediction')
        plt.imshow(depth_c)   
               
        x_img_c = tools.x2image(x_img)
                                                    
        plt.figure(2)
        plt.title('input RGB')
        plt.imshow(x_img_c)    
                  
            
        plt.show()  
            
            
                      
            

   
