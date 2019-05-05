#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np 
import cv2
import torch
import random

def parse_cfg(cfg_file):
    '''
    读取并解析配置文件
    输入： 配置文件目录
    输出： 解析后的结果，list 中的每一个元素为 dict 形式
    '''
    lines = []
    with open(cfg_file, 'r') as f:
        for line in f.readlines():
            if len(line) > 1 and line[0] != '#':
                lines.append(line.strip())
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks

def predict_transform(prediction, dim, anchors, num_classes):
    '''
    
    输入: prediction - [batch_size, number of input channel, height, width]
                       整个网络输出的结果
          dim - 训练图像的大小，一般为 416
          anchors - 类似 [(10, 13), (16, 30), (33, 23)]
          num_classes - 最后分类的类别各数
    '''

    batch_size = prediction.size(0)
    stride =  dim // prediction.size(2)
    # grid_size 在此处其实和 prediction.size(2)是一样的，加这一步主要是
    # 处理无法整除的情况
    grid_size = dim // stride 
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # print(stride)  # 8
    # print(anchors) # [(10, 13), (16, 30), (33, 23)]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # anchor 的尺寸是在原图中的尺寸，所以需要除 stride 变成在 feature map 中的大小
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # 给 bx, by 加上偏移，偏移量就是左上角的坐标
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5: 5+num_classes] = torch.sigmoid((prediction[:,:, 5 : 5+num_classes]))

    # 最终恢复到原图尺寸
    prediction[:,:,:4] *= stride
    
    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    return unique_tensor


def bbox_iou(box1, boxes2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:,0], boxes2[:,1], boxes2[:,2], boxes2[:,3]
    
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


def non_maximum_suppression(prediction, confidence, num_classes, nms_conf = 0.4):
    
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    write = False
    output = 0

    for ind in range(batch_size):
        image_pred = prediction[ind]        

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        non_zero_ind = torch.nonzero(image_pred[:,4])
        
        if len(non_zero_ind) == 0:
            continue
        else:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1,7)    

        img_classes = unique(image_pred_[:,-1])

        for cls in img_classes:

            # 找到所有属于这个类的检测结果
            class_mask_ind = (image_pred_[:,-1] == cls).nonzero().squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)  # 55, 3
            
            # 按照物体存在概率 objectness confidence 进行降序排列
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1]
            # 应用排列顺序到原始列表中
            image_pred_class = image_pred_class[conf_sort_index]
            # 一共有多少个检测到的框
            idx = image_pred_class.size(0)  

            for i in range(idx):

                # 原始实现代码，我不喜欢用 try/catch 替代 if 的工作
                # try:
                #     ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                # except ValueError:
                #     break
            
                # except IndexError:
                #     break

                # 检测第 i 个框与其后边所有框的 iou
                if i+1 > image_pred_class.size(0):
                    break
                else:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])

                # 如果 iou 大于某个临界点的话，那么就移除
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            # 生成一个等长的带有 batch 序号的 tensor
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            
            # 合成一个元组
            seq = batch_ind, image_pred_class
            
            if not write:
                # 拼接结果，至此每个检测结果有8位，第一位是batch的序号，剩下的7个和原来一样
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    return output



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          # Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1))   # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     # Convert to float
    # img_ = torch.autograd.Variable(img_)      # Convert to Variable
    return img_



def letterbox_image(img, out_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = out_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((out_dim[1], out_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    
    return canvas

def prep_image(img, input_dim):
    """
    Prepare image for inputting to the neural network. 
    Returns a Variable 
    """

    # img = cv2.resize(img, (input_dim, input_dim))
    img = letterbox_image(img, (input_dim, input_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def draw_bounding_box(x, results, colors, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img