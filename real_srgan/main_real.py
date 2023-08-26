
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from time import time
from torch.nn import functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class RealSR_VGGNet(SRVGGNetCompact):
    def __init__(self, scale=4):
        super(RealSR_VGGNet, self).__init__(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=scale, act_type='prelu'
        )
    
        self.mod_pad_h, self.mod_pad_w = 0, 0
        self.scale = scale
        self.tile = 0
        self.tile_pad = 10
        self.pre_pad = 10
        self.mod_scale = None

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        output = F.pad(input, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = output.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            output = F.pad(output, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

        return output

    def post_process(self, output: torch.Tensor) -> torch.Tensor:
        if self.mod_scale is not None:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = self.pre_process(x)
        x = super().forward(x)
        x = self.post_process(x)
        return x
    
class RealESR(RRDBNet):
    def __init__(self, scale=4):
        super(RealESR, self).__init__(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
    
        self.mod_pad_h, self.mod_pad_w = 0, 0
        self.scale = scale
        self.tile = 0
        self.tile_pad = 10
        self.pre_pad = 10
        self.mod_scale = None

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        output = F.pad(input, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = output.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            output = F.pad(output, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

        return output

    def post_process(self, output: torch.Tensor) -> torch.Tensor:
        if self.mod_scale is not None:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = self.pre_process(x)
        x = super().forward(x)
        x = self.post_process(x)
        return x

def load_dni(net_a, net_b, dni_weight, key='params'):
    net_a = torch.load(net_a, map_location=torch.device('cpu'))
    net_b = torch.load(net_b, map_location=torch.device('cpu'))
    for k, v_a in net_a[key].items():
        net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
    return net_a

def enhance1(model_path1, model_path2, output_name):
    img = cv2.imread('seiyu.jpg', cv2.IMREAD_UNCHANGED)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    upsampler = RealESRGANer(
        scale=4,
        model_path=[model_path1, model_path2],
        dni_weight=[0.5, 1 - 0.5],
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        gpu_id=None)
    
    output, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(output_name, output)

def enhance2(model_path, output_name):
    img = cv2.imread('seiyu.jpg', cv2.IMREAD_UNCHANGED)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        gpu_id=None)
    
    output, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(output_name, output)

if __name__ == '__main__':
    
    device = torch.device('cpu')
    scale = 4
    # model = RealSR_VGGNet(scale=scale)
    # model = RealESR(scale=scale)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # print(model)
    
    dni = 0.5
    dni_weight = [dni, 1 - dni]
    model_path1 = 'weights/realesr-general-x4v3.pth'
    model_path2 = 'weights/realesr-general-wdn-x4v3.pth'
    model_path3 = 'weights/RealESRGAN_x4plus.pth'
    model_path4 = 'weights/RealESRNet_x4plus.pth'
    loadnet = torch.load(model_path3, map_location=device)
    # loadnet = load_dni(model_path1, model_path2, dni_weight)

    # enhance2(model_path3, 'output2.jpg')

    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    
    model = model.to(device)

    img = Image.open('seiyu.jpg')
    # ow, oh = img.size
    # # img = transforms.Resize((224,224))(img)
    img = transforms.ToTensor()(img)
    
    start = time()
    print('start')

    model.eval()
    with torch.no_grad():
        output = model(img)

    print('end')
    end = time()
    print(round(end - start, 3), 's')

    output = output.squeeze().float().cpu().clamp_(0, 1)
    output = transforms.ToPILImage()(output)
    # output = transforms.Resize((oh*scale,ow*scale), interpolation=transforms.InterpolationMode.LANCZOS)(output)
    output.save('output3.jpg')

    # traced_model = torch.jit.trace(model, image1)

    # scripted_model = torch.jit.script(model)
    # optimized_model = optimize_for_mobile(scripted_model)
    # optimized_model.save('hat_sr.pt')

    # torch.onnx.export(scripted_model,
    #     image1,
    #     "hat_sr.onnx",
    #     input_names = ['input'],
    #     output_names = ['output'],
    #     # dynamic_axes = {'input': {1:'width', 2:'height'}, 'output':{1:'width', 2:'height'}}, 
    #     opset_version = 16,
    # )
