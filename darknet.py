#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import utils



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, m in enumerate(blocks[1:]):
        module = nn.Sequential()

        if m['type'] == 'convolutional':
            activation = m['activation']
            
            # 如果 module字典中存在 batch_normalize 这个键值的话
            if 'batch_normalize' in m:
                batch_normalize = int(m['batch_normalize'])
                bias = False
            else:
                batch_normalize = 0
                bias = True

            filters = int(m['filters'])
            padding = int(m['pad'])
            kernerl_size = int(m['size'])
            stride = int(m['stride'])

            if padding:
                pad = (kernerl_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernerl_size, stride, pad, bias = bias)
            module.add_module("conv_{}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(index), bn)

            # 在 YOLO 中只包含两种： linear and Leaky ReLU
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module('leaky_{}'.format(index), activn)

        elif m['type'] == 'upsample':
            stride = int(m['stride'])
            upsample = nn.Upsample(scale_factor = stride, mode = 'nearest')
            module.add_module("upsample_{}".format(index), upsample)

        elif m['type'] == 'route':
            m['layers'] = m['layers'].split(',')
            
            start = int(m['layers'][0])
            end = 0
            if len(m['layers']) == 2:
                end = int(m['layers'][1])

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif m['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif m['type'] == 'yolo':
            mask = m['mask'].split(',')
            mask = [int(i) for i in mask]

            anchors = m['anchors'].split(',')
            anchors = [int(i) for i in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class DarkNet(nn.Module):
    def __init__(self, cfg_file):
        super(DarkNet, self).__init__()
        self.blocks = utils.parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for index, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(i) for i in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - index

                if len(layers) == 1:
                    x = outputs[index + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - index

                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[index - 1] + outputs[index + from_]

            elif module_type == "yolo":
                anchors = self.module_list[index][0].anchors

                input_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data
                x = utils.predict_transform(x, input_dim, anchors, num_classes)

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[index] = x

        return detections

    def load_offical_weights(self, path):
        '''
        根据路径 path 路径读取权重文件
        '''
        #
        fp = open(path, "rb")
    
        # 前五个值为头文件信息  
        # 1. 主要版本序号
        # 2. 次版本序号
        # 3. 次次版本序号
        # 4,5. 训练图片信息
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
    
            # 如果是卷积层的话就读取权重，否则忽略
            if module_type == "convolutional":
                model = self.module_list[i]

                if 'batch_normalize' in self.blocks[i+1]:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                else:
                    batch_normalize = 0

                conv = model[0]
                
                if batch_normalize:
                    bn = model[1]
        
                    num_bn_biases = bn.bias.numel()
        
                    bn_biases = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr : ptr+num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    num_biases = conv.bias.numel()
                
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    conv.bias.data.copy_(conv_biases)
                    
                num_weights = conv.weight.numel()
                
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)