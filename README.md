# DL_UCAS

UCAS 2018-2019学年深度学习课程实验，包含部分课程实验内容的全部实验代码，分为以下几个部分。

## 图像类

### Experiment1 车牌识别

构造简单的卷积神经网络模型，以实现中国普通机动车车牌字符的识别。

数据集格式：

├─train
│  ├─area
│  ├─letter
│  └─province
└─val
    ├─area
    ├─area
    ├─letter
    └─province

### Experiment2 猫狗分类

对已经在ImageNet数据集上训练好的模型进行微调，实现一个猫狗分类神经网络模型的训练、测试和导出。

数据集格式：

├─train
│  ├─cats
│  └─dogs
└─validation
    ├─cats
    └─dogs

## 语音类

### Experiment3 英文数字语音识别


使用LibROSA包对语音信号进行特征提取。构建卷积神经网络模型，实现英文数字语音zero-nine的识别。

数据集格式：

├─recordings
└─test

### Experiment4 声纹识别

搭建声纹识别模型并训练，实现对未知人的声音进行识别。

数据集格式：

├─test_tisv
└─train_tisv

## 自然语言处理类

### Experiment5 情感分类

建立Text-CNN模型，实现对中文电影评论的情感分类。

数据集格式：

​    info.txt
​    test.txt
​    train.txt
​    validation.txt

数据集下载地址暂未更新，之后会放出。
