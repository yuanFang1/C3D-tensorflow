## 1.数据集

* UCF-11
* UCF-101

## 2. 数据预处理

由于UCF-11和UCF-101的处理方式一致，这里仅以UCF-101为例

1. 将UCF-101解压至videoFile下

 2. 在C3D-tensorflow文件目录下运行video_to_images_ucf101.sh将UCF-101的视频数据按照某个帧率（比如说5）切割成图片，运行命令为：

    `./list/video_to_images_ucf101.sh ./videoFile/UCF-101 5`

3. 在C3D-tensorflow/list文件目录下运行images_to_list_ucf101.py将UCF-101划分成训练集和测试集，并为之添加标签（由于UCF-101一共101类，所以其标签为0-100），运行命令为

   `python images_to_list_ucf101.py`

## 3.使用预训练模型

由于没有强大的计算资源，这里利用了[c3d_ucf101_finetune_whole_iter_20000_TF.model](https://www.dropbox.com/s/u5fxqzks2pkaolx/c3d_ucf101_finetune_whole_iter_20000_TF.model?dl=0)获取别人已经训练好的C3D模型参数进行模型参数的初始化，然后在此基础上使用固定卷积核参数，然后只训练全连接层的方法对模型参数进行微调。



