# import onnx
# import torch
import onnxruntime
import numpy as np
import cv2

if __name__ == '__main__':
    session = onnxruntime.InferenceSession('realesr_net.onnx')

    img = cv2.imread('seiyu.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (np.array(img) / 255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_feed = {input_name: img}
    output = session.run([output_name], input_feed)

    output = output[0].clip(0, 1) * 255
    output = output.astype(np.uint8)
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('seiyu_realesr_net.jpg', output)

