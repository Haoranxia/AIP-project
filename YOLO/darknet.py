import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get rid of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     

        else:
            key, value = line.split("=") 
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    
    return blocks


def create_modules(blocks):
    """
    Creates the modules given list of blocks parsed by parse_cfg
    """
    net_info = blocks[0]                # Network information at pos 0

    module_list = nn.ModuleList()       # List of modules we add to

    prev_filters = 3                    # Input has 3 channels (rgb)
                                        # to be used as depth for next layers' filters

    output_filters = []                 # Helps us keep track of concatenated filter sizes 
                                        # of previous layers brought in by [route] module
                                        # Add to this list so we know the output dimensions per layer
                                        # To be used in the forward function of the network
    
    # Parse (Darknet) config file
    for idx, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Conv layer     
        if block["type"] == "convolutional":
            filters, module = parse_conv(prev_filters, idx, block)

        # Upsampling layer always upsample with stride 2 and bilinearly
        elif block["type"] == "upsample":
            stride = int(block["stride"]) 
            upsample = nn.Upsample(scale_factor=stride, mode="nearest") 
            module.add_module("upsample_{}".format(idx), upsample)
        
        # Route layer is a layer where we concat the specified layers output or just route the specified 
        # layers' output to the next layer
        elif block["type"] == "route":
            filters, module = parse_route(output_filters, idx, block)
        
        # Shortcut layer is a layer where we add the specified layers output to last layers output
        elif block["type"] == "shortcut":
            shortcut_layer = int(block["from"])
            module.add_module("shortcut_{}".format(idx), ShortcutLayer(shortcut_layer))
        
        # YOLO (Detection) layer
        elif block["type"] == "yolo":
            module = parse_yolo(idx, block, net_info)

        module_list.append(module)
        output_filters.append(filters)
        prev_filters = filters

    return net_info, module_list, output_filters


def parse_conv(nr_filters, idx, block):
    """
    Construct [Conv] layer from parsed cfg file
    """
    module = nn.Sequential()

    # Parse params for conv layer
    # Batch_norm present
    try: 
        batch_norm = int(block["batch_normalize"])
        bias = False
    except:
        batch_norm = 0
        bias = True

    filters, size, stride, pad = \
        int(block["filters"]), int(block["size"]), int(block["stride"]), bool(block["pad"])
    
    # Note pad is a boolean. If pad == 1 we add padding as described here:
    # https://github.com/pjreddie/darknet/issues/950
    padding = (size - 1) // 2 if pad else 0

    # Conv layer
    conv = nn.Conv2d(nr_filters, filters, size, stride, padding, bias=bias)
    module.add_module("conv_{0}".format(idx), conv)

    # Batch norm
    if batch_norm:
        batch_norm_layer = nn.BatchNorm2d(filters)
        module.add_module("batch_norm_{0}".format(idx), batch_norm_layer)
    
    # Leaky ReLU activation
    if block["activation"] == "leaky":
        leaky = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leaky_{0}".format(idx), leaky)

    # Return filters to keep track of how many filters we used in this layer
    return filters, module


def parse_route(output_filters, idx, block):
    """
    Construct a [Route] layer from cfg file
    """
    module = nn.Sequential()

    # Parse params for route layer
    # route describes layers as: '-a, b' where -a is -a from current (idx) layer
    # and b is the 'b-th' layer from the start
    # if there are 2 nrs we concat layers: 'idx-a' and 'b'
    
    # Parse and cast to int
    layers = [int(layer) for layer in block["layers"].split(',')]

    route = RouteLayer(layers)
    module.add_module("route_{0}".format(idx), route)

    # only -a
    if len(layers) == 1:
        return output_filters[idx + layers[0]], module

    # '-a' and 'b' -> we concat output of '-a' and 'b'
    else:
        return output_filters[idx + layers[0]] + output_filters[layers[1]], module


def parse_yolo(idx, block, net_info):
    module = nn.Sequential()

    mask = [int(x) for x in block["mask"].split(',')]   # used to determine with anchors to take
    anchors = [int(x) for x in block["anchors"].split(',')]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    input_dim = (int(net_info["height"]), int(net_info["width"]))
    num_classes = int(block["classes"])

    detection = DetectionLayer(anchors, input_dim, num_classes)
    module.add_module("Detection_{0}".format(idx), detection)

    return module





"""
Actual Darknet Model
"""
class Darknet(nn.Module):
    def __init__(self, cfg):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg)
        self.net_info, self.module_list, self.filters  = create_modules(self.blocks)

    
    def forward(self, x, CUDA):
        """
        Iterate over blocks (modules) and perform forward pass
        """
        detections = None
        modules = self.blocks[1:]   # modules start 
        outputs = {}                # cache outputs for 'route' and 'shortcut' layer

        # Perform forward pass for every module
        for idx in range(len(modules)):
            module_type = modules[idx]["type"]

            # Forward pass logic for conv and upsample layers
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[idx](x)
                #outputs[idx] = x

            # Forward pass logic for yolo layer
            elif module_type == "yolo":
                x = self.module_list[idx](x)

                # Make predictions. Concatenate predictions if we already
                # have made some before. We want to concat the predictions
                # accross different detection scales hence this part
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
                
                #outputs[idx] = outputs[idx - 1]

            # Forward pass logic for Shortcut, Route layers
            else:
                self.module_list[idx][0].set_values(outputs, idx)
                x = self.module_list[idx](x)
            
            outputs[idx] = x

        return detections


    def load_weights(self, weightfile):
        """
        Function that loads weights from a weights file
        """
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
