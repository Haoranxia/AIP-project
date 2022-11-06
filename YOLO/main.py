from darknet import *
import torch
import cv2
import os 
import os.path as osp

from util import *

# cfg and weight file
cfg_file = './yolov3.cfg'
weights_file = '../data/yolov3.weights'
names_file = '../data/coco.names'

# Images path
im_dir = "../data/testing_images/"

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
    img = img[:, :, :].transpose((2,0,1)).copy()
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

    try:
        img_list = [osp.join(osp.realpath('.'), im_dir, img) for img in os.listdir(im_dir)]
    except Exception:
        print("Image dir not found")
        return 

    # Prepare images
    images = [cv2.imread(img) for img in img_list]
    inp_dim = 608 # TODO
    prepped_images = [prep_image(img, inp_dim) for img in images]

    final_results = []
    for idx in range(len(prepped_images)):
        # Images in question
        prepped_image = prepped_images[idx]
        real_image = images[idx]

        # Run model
        prediction = model(prepped_image)
        output = format_output(prediction, confidence=0.5, nms_conf=0.3)

        # Rescale boxes to fit original images
        orig_w, orig_h = images[idx].shape[1], images[idx].shape[0]
        scale_w, scale_h = orig_w / inp_dim, orig_h / inp_dim
        output[:, 1] *= scale_w 
        output[:, 2] *= scale_h
        output[:, 3] *= scale_w 
        output[:, 4] *= scale_h

        # Clip boxes so it stays within the image
        output[:, 1] = torch.clamp(output[:, 1], min=25, max=orig_w - 10)
        output[:, 2] = torch.clamp(output[:, 2], min=25, max=orig_h - 10)
        output[:, 3] = torch.clamp(output[:, 3], min=10, max=orig_w - 10)
        output[:, 4] = torch.clamp(output[:, 4], min=10, max=orig_h - 10)

        final_results.append(draw_bboxs(real_image, output, classes))
    
    # Save images with bboxes 
    for idx in range(len(final_results)):
        cv2.imwrite("../output/out_{}.png".format(idx), final_results[idx])

main()











