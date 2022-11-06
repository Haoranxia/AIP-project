from darknet import *
import torch
from torch.autograd import Variable
import cv2
import os 
import os.path as osp

from util import *

# cfg and weight file
cfg_file = './yolov3.cfg'
weights_file = '../data/yolov3.weights'
names_file = '../data/coco.names'

# Testing func
def get_test_input(im_path):
    try:
        img = cv2.imread(im_path)
        img = cv2.resize(img, (608, 608))                                    # Resize to the input dimension
        img_ = img[:, :, ::-1].transpose((2, 0, 1))                          # BGR -> RGB | H X W C -> C X H X W 
        img_ = img_[np.newaxis, :, :, :] / 255.0                             # Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()                                # Convert to float
        img_ = Variable(img_)                                                # Convert to Variable
        return img_

    except Exception as e: 
        print("Something went wrong: ", e)
        return None


def draw_bboxs(image, bboxs, classes):
    """
    image:      image
    bboxs:      2D Tensor of format [[image_idx, x1, y1, x2, y2, class_idx], ...]
    """
    for bbox in bboxs:
        x1, y1 = int(bbox[1].item()), int(bbox[2].item())
        x2, y2 = int(bbox[3].item()), int(bbox[4].item())

        label = classes[int(bbox[5])]
        label_x, label_y = x1, y1 - 10

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (label_x, label_y), 0, 0.3, (0, 255, 0))
    
    return image

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img


def main():
    # Model
    model = Darknet(cfg_file)
    model.load_weights(weights_file)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    classes = load_classes(names_file)

    # Image path
    im_dir = "../data/testing_images"

    try:
        img_list = [osp.join(osp.realpath('.'), im_dir, img) for img in os.listdir(im_dir)]
    except Exception:
        print("Image dir not found")
        return 

    images = [cv2.imread(img) for img in img_list]
    inp_dim = 608 # TODO: Change this to get from cfg file
    prepped_images = [prep_image(img, inp_dim) for img in images]

    final_results = []
    for idx in range(len(prepped_images)):
        # Images in question
        prepped_image = prepped_images[idx]
        real_image = images[idx]

        # Run model
        prediction = model(prepped_image, torch.cuda.is_available())
        output = format_output(prediction, confidence=0.5, nms_conf=0.3)

        # Rescale boxes to fit original images
        orig_w, orig_h = images[idx].shape[1], images[idx].shape[0]
        scale_w, scale_h = orig_w / inp_dim, orig_h / inp_dim
        output[:, 1] *= scale_w 
        output[:, 2] *= scale_h
        output[:, 3] *= scale_w 
        output[:, 4] *= scale_h

        final_results.append(draw_bboxs(real_image, output, classes))
    
    # Save images in final results
    for idx in range(len(final_results)):
        cv2.imwrite("../output/out_{}.png".format(idx), final_results[idx])


main()

# inp = get_test_input(im_path)
# inp2 = get_test_input(im_path)
# inp = torch.cat((inp, inp2), 0)

# Testing
# if inp is not None:
#     pred = model(inp, torch.cuda.is_available())

#     output = format_output(pred, confidence=0.5, nms_conf=0.3)
#     print("output size: ", output.size())
#     print(output)

#     print(type(inp))
#     draw_bboxs(inp, output, classes)











