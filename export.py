import torch
from main_real import RealESR, RealSR_VGGNet, load_dni

if __name__ == '__main__':
    device = torch.device('cpu')

    model = RealESR(scale=4)
    # model = RealSR_VGGNet(scale=4)

    dni = 0.5
    dni_weight = [dni, 1 - dni]
    model_path1 = 'real_srgan/weights/realesr-general-x4v3.pth'
    model_path2 = 'real_srgan/weights/realesr-general-wdn-x4v3.pth'
    model_path3 = 'real_srgan/weights/RealESRGAN_x4plus.pth'
    model_path4 = 'real_srgan/weights/RealESRNet_x4plus.pth'

    loadnet = torch.load(model_path4, map_location=device)
    # loadnet = load_dni(model_path1, model_path2, dni_weight)

    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.train(False)
    model.cpu().eval()
    
    rand_img = torch.rand(1, 3, 240, 240)

    traced_model = torch.jit.trace(model, rand_img)

    with torch.no_grad():
        scripted_model = torch.jit.script(traced_model)
        
        # optimized_model = optimize_for_mobile(scripted_model)
        # optimized_model.save('realesr_net.pt')
        
        torch.onnx.export(scripted_model,
            rand_img,
            "realesr_net.onnx",
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {'input': {2:'width', 3:'height'}, 'output':{2:'width', 3:'height'}}, 
            opset_version = 16,
        )

    
    # traced_model = torch.jit.trace(model, image)

    # scripted_model = torch.jit.script(model)
    # optimized_model = optimize_for_mobile(scripted_model)
    # optimized_model.save('hat_sr.pt')
    # model.train(False)
    # model.cpu().eval()
    # with torch.no_grad():
    #     torch.onnx.export(model,
    #         image,
    #         "hat_sr.onnx",
    #         input_names = ['input'],
    #         output_names = ['output'],
    #         # dynamic_axes = {'input': {1:'width', 2:'height'}, 'output':{1:'width', 2:'height'}}, 
    #         opset_version = 16,
    #     )