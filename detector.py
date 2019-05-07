#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils
import os
import cv2
import time
import torch
import argparse

import numpy as np 
import pickle as pkl
from darknet import DarkNet

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", help = "Image / Directory containing images to perform detection upon",
                        default = "images", type = str)
    parser.add_argument("--det", help = "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--batch_size", help = "Batch size", default = 1)
    parser.add_argument("--confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", help = "Config file", default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights",  help = "weights file", default = "yolov3.weights", type = str)
    parser.add_argument("--resolution", help = "Input resolution of the network.",
                        default = "416", type = str)
    return parser.parse_args()


args = arg_parse()

# 读取每个类别对应的名字
num_classes = 80
classes = utils.load_classes("data/coco.names")

# 读取网络，并加载训练好的网络参数
print("Loading network.....")
model = DarkNet(args.cfg)
model.load_offical_weights(args.weights)
print("Network successfully loaded")

model.net_info["height"] = args.resolution
input_dim = int(args.resolution)


# 设置成 evaluation 模式
model.eval()

# 给定文件夹，读取需要预测的图片
if not os.path.exists(args.images):
    print ("No file or directory with the name {}".format(args.images))
    exit()
else:
    imlist = [os.path.join(os.path.realpath('.'), args.images, img) for img in os.listdir(args.images)]

if not os.path.exists(args.det):
    os.makedirs(args.det)

loaded_ims = [cv2.imread(x) for x in imlist]
im_batches = list(map(utils.prep_image, loaded_ims, [input_dim for x in range(len(imlist))]))
# print(im_batches[0].shape, 'im_batches[0]')
# torch.Size([1, 3, 416, 416]) im_batches[0]

im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
# print(im_dim_list)
# tensor([[602., 452., 602., 452.],
#         [602., 452., 602., 452.]])


if (len(im_dim_list) % args.batch_size):
    num_batches = len(imlist) // args.batch_size + 1            
    im_batches = [torch.cat((im_batches[i*args.batch_size : min((i+1)*args.batch_size, len(im_batches))]))
                 for i in range(num_batches)]  

output = 0
for i, batch in enumerate(im_batches):
    start = time.time()
    with torch.no_grad():
        prediction = model(batch)

    # print(prediction.shape, 'prediction')
    # torch.Size([1, 10647, 85])

    prediction = utils.non_maximum_suppression(prediction, args.confidence, num_classes, nms_conf = args.nms_thresh)
    end = time.time()

    if isinstance(prediction, int):
        for im_num, image in enumerate(imlist[i*args.batch_size: min((i +  1)*args.batch_size, len(imlist))]):
            im_id = i*args.batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/args.batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * args.batch_size

    if isinstance(output, int):
        output = prediction
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * args.batch_size : min((i + 1) * args.batch_size, len(imlist))]):
        im_id = i * args.batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / args.batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")      

if isinstance(output, int):
    print("No detections were found.")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1,1)

output[:,[1,3]] -= (input_dim - scaling_factor * im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (input_dim - scaling_factor * im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
# 以上处理后的 output 的检测框的坐标就是在原始未 resize 的图像中的坐标

colors = pkl.load(open("pallete", "rb"))

# 在原始图像上画图像并且保存到指定文件夹
list(map(lambda x: utils.draw_bounding_box(x, loaded_ims, colors, classes), output))
names = [os.path.join(args.det, 'det_' + os.path.basename(i)) for i in imlist]
list(map(cv2.imwrite, names, loaded_ims))