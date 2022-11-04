import torch
import torch.nn as nn

"""
File containing utility functions and custom layers
"""
class ShortcutLayer(nn.Module):
    """
    Models a [Shortcut] Layer as described by the yolov3.cfg file
    """
    def __init__(self, shortcut_from):
        super(ShortcutLayer, self).__init__()
        self.shortcut_from = shortcut_from          # offset idx from current idx in module_list
        self.outputs = None
        self.idx = None

    def forward(self, x):
        """
        x:          input
        outputs:    list of cached outputs
        idx:        idx of current module 
        """

        # Defined as a skip layer basically
        x = self.outputs[self.idx - 1] + self.outputs[self.idx + self.shortcut_from]
        return x

    def set_values(self, outputs, idx):
        self.outputs = outputs
        self.idx = idx



class RouteLayer(nn.Module):
    """
    Models a [Route] Layer as described by the yolov3.cfg file
    """
    def __init__(self, layers):
        super(RouteLayer, self).__init__()
        self.layers = layers                        # offset idx, and possibly route idx
                                                    # contains [-a, b] where -a is offset from current module idx
                                                    # b is idx of module to concat with
        self.outputs = None
        self.idx = None

    def forward(self, x):
        """
        x:          input
        outputs:    list of cached outputs
        idx:        idx of current module 
        """

        # (idx + -a)th layer
        x = self.outputs[self.idx + self.layers[0]]

        # b-th layer
        if len(self.layers) > 1:
            y = self.outputs[self.layers[1]]
            x = torch.cat((x, y), 1)

        return x
    
    def set_values(self, outputs, idx):
        self.outputs = outputs
        self.idx = idx



class DetectionLayer(nn.Module):
    """
    Models a [Detection] or YOLO Layer as described by the yolov3.cfg file
    """
    def __init__(self, anchors, input_dim, num_classes):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors                      # anchor points
        self.input_dim = input_dim
        self.num_classes = num_classes


    def forward(self, x):
        """
        Outputs a tensor containing:
        - 0 or 1 whether obj has been detection
        - bbox attributes
        - class predictions
        """
        predictions = self.predict_cells(x, self.input_dim, self.num_classes)
        return predictions


    def predict_cells(self, x, input_dim, num_classes, CUDA = False):
        """
        For each cell, construct a prediction vector size of (B * (5 + C))
        Basically a 2D Tensor of size: (cell * cell) x (B * (5 + C))
        Repeat this for each batch: (B, c*c, b*(5+c)) output

        Note that the output of the layer before this (input x) defines how many grid cells we 
        will have. Since each feature map will have some receptive field of the original input

        x:              conv network output of input image as feature map of (b, c, h, w)
                        (h,w) is the result of a bunch of 1x1 convs
        input_dim:      (H, W) of input image
        num_classes:    nr. of classes to detect
        """
    
        stride = input_dim[0] // x.size(2)          # Our feature maps are [x.size(2) * x.size(3)]
                                                    # Each of these feature map cells has a receptive field encompassing
                                                    # a portion of the input. We use this receptive field to define our
                                                    # grid sizes

        grid_size = input_dim[0] // stride          # Size of each grid. The nr of grids we have is input dimensions / stride

        bbox_attributes = 5 + num_classes           # Obj detected, x, y, w, h, num_classes => 5 + num_classes
        num_anchors = len(self.anchors)

        # Reshape to (B, b*(5+c)*anchors, c*c) because our data is structured in such a way that
        # the c*c data is in the last 2 dimensions. Alsot ake nr anchors into account
        x = x.view(x.size(0), bbox_attributes * num_anchors, grid_size * grid_size)

        # Transpose 1st and 2nd dim so we can obtain desired output dimensions
        x = x.transpose(1, 2).contiguous()

        # Reshape to get desired result of (B, c*c*anchors, b*(5+c)) since we want
        # the bbox attributes to be in the last dimension
        x = x.view(x.size(0), grid_size * grid_size * num_anchors, bbox_attributes)

        # Rescale anchors so they fit into the cell
        anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in self.anchors]

        # Sigmoid (x,y) coords and obj confidence
        # Note output per cell y = [bbox params + object confidence + class confidences] [4 + 1 + num_classes]
        # By definition yolov3 predicts sigmoid(x) sigmoid(y) as output
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])

        # Evenly spaced values [0, grid_size) of 2d coordinates (x,y)
        grid = torch.arange(grid_size)
        
        # Compute offsets for bounding box (x,y) coordinates
        offsets = torch.FloatTensor([[b, a] for a in grid for b in grid])

        # Repeat for each anchor box and reshape it into desired output
        # output desired is a tensor of [[offset_x0_box0, offet_yo_box0], ...]
        offsets = offsets.repeat(1, num_anchors).view(-1, 2)

        # Reshape it into a correct dimension for adding it to the output predictions
        offsets = offsets.unsqueeze(0)

        # Add offsets to bounding box (x,y) parameters 
        x[:, :, :2] += offsets

        # Perform similar calculations for anchor boxes
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)

        # yolov3 predicts exp(w), exp(h) for bbox w,h
        x[:, :, 2:4] = anchors * torch.exp(x[:, :, 2:4])

        # Softmax the class prediction scores
        x[:, :, 5:5 + num_classes] = torch.sigmoid((x[:, :, 5:5 + num_classes]))

        # Resize YOLO detection map outputs to size of input image
        x[:, :, :4] *= stride

        # Note that for ground truth comparison we should invert the prediction format for
        # the bbox (x,y) and (w,h) 
        # x, y = sigmoid_inverse(x, y)
        # w, h = ln(w, h) 
        return x



def format_output(prediction, confidence, num_classes, nms_conf=0.5):
    """
    Write prediction volume (output of YOLOv3 network) to a format that
    is better to understand

    prediction:     Tensor of (B, num_bbox's, 5 + num_classes) 
    confidence:     Confidence threshold for keeping a bbox
    nms_conf:       Confidence threshold for non max suppression
    """

    # Remove low confidence bbox's
    # Note that in prediction[:, :, 4] lies the confidence of object detection
    # for that specific bounding box
    # 
    # print(prediction.size())
    # prediction = prediction[prediction[:, :, 4] > confidence]
    # print("size: ", prediction.size())

    # conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    # prediction = prediction * conf_mask

    # Reformat bbox params (center_x, center_y) and (w, h) 
    # to (topleft_x, topleft_y, botright_x, botright_y)
    bounding_boxes = torch.clone(prediction)
    bounding_boxes[:, :, 0] = prediction[:, :, 0] - (prediction[:, :, 2] / 2)       # center_x - half width
    bounding_boxes[:, :, 1] = prediction[:, :, 1] - (prediction[:, :, 3] / 2)
    bounding_boxes[:, :, 2] = prediction[:, :, 0] + (prediction[:, :, 2] / 2)
    bounding_boxes[:, :, 3] = prediction[:, :, 1] + (prediction[:, :, 3] / 2)

    # For every image (in our batch) find final bbox's
    for idx in range(prediction.size(0)):
        print("\nnew batch")
        # Input of size: (nr_bbox_predictions, 5 + num_classes)
        image_prediction = prediction[idx]

        # Confidence thresholding
        # Remove all predictions (bbox's) with low confidence
        # Tensor size: (tresholded_nr_bbox, 5 + num_classes)
        image_prediction = image_prediction[image_prediction[:, 4] > confidence]

        # Non max suppression
        # max_conf_tensor:  tensor of maximum class confidence values per bbox
        # max_conf_indices: tensor of class indices whose max confidence values were found
        #                   note that indices start from 5 + idx
        max_conf_tensor, max_conf_indices = torch.max(image_prediction[:, 5:], 1)
        confidences = torch.stack((max_conf_tensor, max_conf_indices), 1)

        # Concat idx to confidence tensor
        idx_tensor = torch.Tensor([i for i in range(image_prediction.size(0))]).unsqueeze(1)
        confidences = torch.cat((confidences[:, ], idx_tensor), 1)

        # Unique classes identified by detector: 1D Tensor
        unique_classes = torch.unique(max_conf_indices)

        # For each unique class retain perform NMS
        final_bboxes = {}
        for class_ in unique_classes:
            final_bboxes[class_] = non_max_suppression(idx, bounding_boxes, image_prediction, confidences, class_, nms_conf)

            pass

        # print(image_prediction[:, :5].size(), max_conf_tensor.size(), max_conf_indices.size())
        # image_prediction = torch.cat(seq, 0)

        # print(image_prediction)

        # print(max_confidence.size())
        # print(max_confidence)

        # print(max_confidence_score.size())
        # print(max_confidence_score)

        # Obtain classes found in this image


def non_max_suppression(im_idx, bboxs, predictions, confidences, class_idx, nms_conf=0.5):
    """
    im_idx:         The idx of the current image out of all images in our batch
    bboxs:          The bounding boxes found by our detector                             - size() = (batch_size, num_bboxs, 5 + num_classes)
    predictions:    Confidence filtered bbox predictions                                 - size() = (num_bboxs, 5 + num_classes)
    confidences:    2D Tensor of [[conf, class_idx, idx (in predictions)], ...]          - size() = (num_bboxs, 3)
    class_idx:      The class idx (5 + idx in predictions)
    nms_conf:       Confidence threshold for non max suppression
    
    return:         Tensor with non suppressed bboxes for class 'class_idx'
    """
    # Construct mask to filter through predictions which dont match our class
    class_mask = (confidences[:, 1] == class_idx)

    # Apply mask
    matching_predictions = predictions[class_mask, :]       # result: filtered tensor of size (matching_classes, 5 + num_classes)
    matching_confidences = confidences[class_mask, :]       # result: filtered tensor of size (matching_classes, 3)

    # TODO: Change code here to do it on the tensors immediately instead of singling
    #       out tensors and removing them
    
    # Find highest confidence detection for specified class
    highest_conf_prediction_idx = torch.argmax(matching_confidences[:, 0])
    highest_conf_prediction = matching_predictions[highest_conf_prediction_idx]

    # Remove highest conf prediction from matching tensors
    matching_predictions = torch.cat(matching_predictions[:highest_conf_prediction_idx], matching_predictions[highest_conf_prediction_idx:])
    matching_confidences = torch.cat(matching_confidences[:highest_conf_prediction_idx], matching_confidences[highest_conf_prediction_idx:])

    # Add highest confidence prediction to final output
    final_output = torch.Tensor(highest_conf_prediction)

    # For every other detecetion for our class perform NMS with current_bbox
    while matching_predictions.numel() > 0:

        # Iterate over all (remaining) matching predictions
        for idx in range(matching_predictions.size(0)):
            bbox1 = matching_predictions[highest_conf_prediction_idx]
            bbox2 = matching_predictions[idx]
            iou = bbox_IoU(bbox1, bbox2) 

            # If iou is above threshold we remove from the matching tensors
            # they can be suppressed
            if (iou > nms_conf):
                matching_predictions = torch.cat(matching_predictions[:, :idx], matching_predictions[:, idx:])
                matching_confidences = torch.cat(matching_confidences[:, :idx], matching_confidences[:, idx:])
        
        # Assign new prediction to current highest conf prediction and repeat
        highest_conf_prediction_idx = torch.argmax(matching_confidences[:, 0])
        highest_conf_prediction = matching_predictions[highest_conf_prediction_idx]
        final_output = torch.cat((final_output, highest_conf_prediction), 1)

    return final_output


def bbox_IoU(box1, box2):
    """
    Returns IoU of 2 boxes

    box1:       2D Tensor of (nr_bboxs, 5 + num_classes)
    box2:       2D Tensor of (nr_bboxs, 5 + num_classes)

    return:     IoU value between box1, box2
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

     

"""
MISC Code for possible later use
"""
    # # Obtain indices of bbox's that match our class_idx
    # matching_classes = confidences[confidences[:, 1] == class_idx]
    # print("match class size: ", matching_classes.size())
    # print("indices:", matching_classes)

    # # Find maximum confidence in matching confidences
    # max_confidence_idx = torch.argmax(matching_classes[:, 0])
    
    # # For every other bbox of same class suppress it if there is a high IoU
    # non_suppressed_bboxs = bboxs[:,  max_confidence_idx, :]
    # print("non suppr size: ", non_suppressed_bboxs.size())
    # print("non suppr: ", non_suppressed_bboxs)

    # highest_conf_bbox = matching_classes[max_confidence_idx]

    # # Perform NMS between highest_conf_bbox and every other bbox of same class
    # for idx in range(matching_classes.size(0)):
    #     other_bbox = matching_classes[idx]

    #     # We are checking the same bbox as the max confidence bbox
    #     if other_bbox[2] == max_confidence_idx:
    #         continue
        
    #     # Find IoU between highest conf bbox and other bboxs of same class
    #     iou = bbox_IoU(bboxs[:, max_confidence_idx, :], bboxs[:, idx, :])

    #     # Don't suppress if iou is acceptable
    #     if (iou < nms_conf):
    #         non_suppressed_bboxs = torch.cat((non_suppressed_bboxs, bboxs[:, idx, :]))

        

    # return non_suppressed_bboxs





        


        



    
    







