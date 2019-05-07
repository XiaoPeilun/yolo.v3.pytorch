# yolo.v3.pytorch

比较简单的 YOLO v3 实现，有中文注释，在 Python 3 和 PyTorch 0.4 以上版本上应该都可以运行。 只支持 evaluation，即给定一组测试图片，输出检测结果。下载 YOLO 作者提供的 [权重文件](https://pjreddie.com/media/files/yolov3.weights) 放在主文件夹下即可运行，权重是在 coco 数据集上训练得到的，支持80个类别的检测。不支持在自己的数据集上重新训练。只支持 CPU，运行效率不是很高，但是对深入了解 YOLO 的实现有挺大的帮助。



注：该项目是参考 [这个教程](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) 实现的，感谢作者提供如此通俗易懂的教程！


