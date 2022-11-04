from darknet import *
import torch
from torch.autograd import Variable
import cv2

from util import *

cfg_file = './yolov3.cfg'
weights_file = '../data/yolov3.weights'
# blocks = parse_cfg(cfg_file)

# print(len(blocks))
# for block in blocks:
#     print(block)

# net_info, module_list, filter_list = create_modules(blocks)

def get_test_input(im_path):
    try:
        img = cv2.imread(im_path)
        img = cv2.resize(img, (608, 608))                                    # Resize to the input dimension
        img_ = img[:,:,::-1].transpose((2, 0, 1))                           # BGR -> RGB | H X W C -> C X H X W 
        img_ = img_[np.newaxis, :, :, :] / 255.0                             # Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()                                # Convert to float
        img_ = Variable(img_)                                                # Convert to Variable
        return img_

    except Exception as e: 
        print("Something went wrong: ", e)
        return None

model = Darknet(cfg_file)
model.load_weights(weights_file)
im_path = "../data/dog-cycle-car.png"

inp = get_test_input(im_path)
inp2 = get_test_input(im_path)

inp = torch.cat((inp, inp2), 0)

if inp is not None:
    pred = model(inp, torch.cuda.is_available())
    format_output(pred, 0.4, 80)



