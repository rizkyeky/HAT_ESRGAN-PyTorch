
from copy import deepcopy
import sys
sys.path.append('hat_sr')

# sys.argv.extend(['-opt', 'hat_sr/options/test/HAT_SRx4.yml'])

from hat_sr.hat.archs.hat_arch import HAT
# import hat.hat.data
# import hat_sr.hat.models
# import os.path as osp
# from basicsr.models import build_model
# from basicsr.models.sr_model import SRModel
# from basicsr.archs import build_network
# from basicsr.utils.options import dict2str, parse_options
import torch
import torchvision.transforms as transforms
from PIL import Image
from time import time
from torch.nn import functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

class HAT_SR(HAT):
    def __init__(self, scale=4):
        super(HAT_SR, self).__init__(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )
    
        self.mod_pad_h, self.mod_pad_w = 0, 0
        self.scale = scale

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = input.size()
        if h % self.window_size != 0:
            self.mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            self.mod_pad_w = self.window_size - w % self.window_size
        output = F.pad(input, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
        return output

    def post_process(self, output: torch.Tensor) -> torch.Tensor:
        _, _, h, w = output.size()
        return output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_process(x)
        x = super().forward(x)
        x = self.post_process(x)
        return x

class HAT_S_SR(HAT_SR):
    def __init__(self):
        super(HAT_S_SR, self).__init__(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=24,
            squeeze_factor=24,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=144,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

def load_net_hat_sr(model_path: str):
    loadnet = torch.load(model_path)
    loadnet = loadnet['params_ema']
    for k, v in deepcopy(loadnet).items():
        if k.startswith('module.'):
            loadnet[k[7:]] = v
            loadnet.pop(k)
    return loadnet

if __name__ == '__main__':
    
    device = torch.device('cpu')
    
    model = HAT_SR()
   
    model_path = 'hat_sr/experiments/pretrained_models/Real_HAT_GAN_SRx4.pth'
    loadnet = load_net_hat_sr(model_path)
    model.load_state_dict(loadnet, strict=True)

    model = model.to(device)

    image1 = Image.open('naruto.jpg')
    ow, oh = image1.size
    # image1 = transforms.Resize((224,224))(image1)
    image1 = transforms.ToTensor()(image1)
    image1 = image1.unsqueeze(0)

    start = time()
    model.eval()
    with torch.no_grad():
        print('start')
        output = model(image1)
        print('end')

    end = time()
    print(round(end - start, 3), 's')

    output1 = output.squeeze(0)
    output1 = transforms.ToPILImage()(output1)
    # output1 = transforms.Resize((oh*4,ow*4))(output1)
    output1.save('output2.jpg')

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
