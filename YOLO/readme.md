# YOLOv3 implementation
Implementation of the YOLOv3 algorithm. The network in the paper is Darknet53 (https://github.com/pjreddie/darknet) which is used for this implementation. 

The network adheres to the architecture as described in (https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

Pre-trained weights for the network are also obtained from the authors here (https://pjreddie.com/darknet/yolo/)

## predict_cells() in util.py
This function takes the network output (pre-trained network that outputs a feature map where each cell of the last 2 dimensions correspond to a certain receptive field of the original input)

Our input is a volume of (batch_size, channels/features, h, w) 
Our desired output is a volume of (batch_size, nr_grids, attributes)

We must thus reshape the input into the desired volume for the sake of processing the data more easily. This is done in the first part of this function

Another thing is that we must rescale and offset the output and anchor boxes to coordinates that match our input image.

## format_output() in util.py
This function takes the output from predict_cells() and formats it in a way that is ready to be drawn on the input image.

We first go from (x_center, y_center, h, w) parameters for our bounding boxes to (x1, y1, x2, y2).

We then loop over every item in our batch (every image) and filter out bounding boxes that are not nessecary.

We first filter by removing all bounding boxes with a low confidence of object detection.

We the perform non max suppression per detected class of objects to filter out remaining bounding boxes until only one is left per detected object.

Finally our output is formatted as a 2D tensor of size:(bounding_boxes, box_parameters) where box_parameters refer to the values [x1, y1, x2, y2]


## non_max_suppression() in util.py
This function takes in a tensor of bounding boxes (bounding_boxes, box_params) which is sorted in a descending order by class confidence. 

For every box we perform non max suppression of the currently indexed box, and all other remaining boxes. We compute the IoU of these values and threshold based on our nms_conf parameter. If the threshold is below this parameter we keep this box and move on to the next of the remaining boxes.
If this is not the case we get rid of the boxes that exceed our threshold and suppress them in the final output


## bbox_IoU in util.py 
This function takes in a 2 volumes of bounding boxes 
- box1:     1 bounding box, the box we check against 
- box2:     n bounding boxes, a volume of boxes 

We then compute the Intersection and Union of these boxes and return the Intersection over Union


## Parsing yolov3 config file into a Pytorch Model in darknet.py
To construct the network and to be able to load in the pre-trained weights we have to adhere to the architecture as described by the authors. 

The architecture can be found in a yolov3.cfg file that is found on their github repository. This file contains descriptions of all the modules and its parameters that we parse and construct a network out of.

The weights file are done in a similar fashion where the values in the weights file map one to one to the architecture in terms of weights <-> module. Parsing the weights file and loading it into the network was done using a method I found online.

We also had to make custom forward functions for some of the modules. Details can be found in the respective classes in 'util.py'

# How to use
Simply run main.py with all the file paths set correctly. Files required are as follows:

- **yolov3.cfg:**       contains a description of the network architecture
- **yolov3.weights:**   contains a bunch of pre-trained weights for the yolov3 network
- **coco.names:**      the network is trained on the COCO dataset so a list of object names for images in the COCO dataset
- **image_directory:**  directory of where the testing images reside

Requires **Pytorch, cv2, numpy** libraries
