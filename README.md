# paddleOCR2
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 介绍了如何完成PaddleOCR中文字检测模型的训练、评估与测试，描述了部分程序原理，以及我在实现过程中遇到的一些报错。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PaddleOCR使用了DB文本检测算法，它支持MobileNetV3、ResNet50_vd两种骨干网络，可以根据需要修改相应的配置文件。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 在本次repo中使用的是MobileNetV3作为骨干网络，使用icdar15数据集进行训练。

## 1、数据集的准备
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 本次使用的数据集是icdar2015数据集，数据集的下载地址为：https://rrc.cvc.uab.es/?ch=4&com=downloads

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 下载完成后，将数据集解压到icdar2015文件夹下，文件夹的目录结构如下：
```
icdar2015/text_localization 
  └─ icdar_c4_train_imgs/         icdar数据集的训练数据
  └─ ch4_test_images/             icdar数据集的测试数据
  └─ train_icdar2015_label.txt    icdar数据集的训练标注
  └─ test_icdar2015_label.txt     icdar数据集的测试标注
 ```
标注文件的格式如下：
```
ch4_test_images/img_424.jpg	[{"transcription": "SAMSUNG", "points": [[701, 128], [837, 39], [843, 104], [707, 193]]}, {"transcription": "###", "points": [[299, 304], [325, 304], [325, 314], [298, 314]]}, {"transcription": "###", "points": [[325, 300], [351, 302], [351, 314], [325, 312]]}, {"transcription": "Tokyo", "points": [[353, 299], [385, 299], [382, 313], [350, 313]]}, {"transcription": "###", "points": [[1005, 230], [1025, 230], [1025, 241], [1005, 241]]}]
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; json.dumps编码前的图像标注信息是包含多个字典的list，字典中的points表示文本框的四个点的坐标(x, y)，从左上角的点开始顺时针排列。 transcription中的字段表示当前文本框的文字，在文本检测任务中并不需要这个信息。 如果您想在其他数据集上训练PaddleOCR，可以按照上述形式构建标注文件。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 如果"transcription"字段的文字为'*'或者'###‘，表示对应的标注可以被忽略掉，因此，如果没有文字标签，可以将transcription字段设置为空字符串。

## 2、数据预处理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 数据预处理的代码在`https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.4/ppocr/data/imaug/operators.py` 中

### 图像和标签的解码

在图像和标签的解码阶段遇到了一个警告如下：
```
D:\mycode\paddleOCR2\decode.py:74: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  txt_tags = np.array(txt_tags, dtype=np.bool)
```
这是因为代码中使用了np.bool，而np.bool是一个已经被弃用的别名，建议使用bool来替代。如果确实需要使用NumPy的布尔类型，可以使用np.bool_。
这个警告是从NumPy 1.20版本开始引入的。

### 基础数据增广

数据增广是提高模型训练精度，增加模型泛化性的常用方法，文本检测常用的数据增广包括随机水平翻转、随机旋转、随机缩放以及随机裁剪等等。

随机水平翻转、随机旋转、随机缩放的代码实现参考代码。随机裁剪的数据增广[代码](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.4/ppocr/data/imaug/iaa_augment.py)实现参考[代码](https://github.com/PaddlePaddle/PaddleOCR/blob/81ee76ad7f9ff534a0ae5439d2a5259c4263993c/ppocr/data/imaug/random_crop_data.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L127)。

### 获取阈值图标签

使用扩张的方式获取算法训练需要的阈值图标签，详细[代码](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.4/ppocr/data/imaug/make_border_map.py)参考。

### 获取概率图标签

使用收缩的方式获取算法训练需要的概率图标签；

### 归一化

通过规范化手段，把神经网络每层中任意神经元的输入值分布改变成均值为0，方差为1的标准正太分布，使得最优解的寻优过程明显会变得平缓，训练过程更容易收敛；

### 通道变换

图像的数据格式为[H, W, C]（即高度、宽度和通道数），而神经网络使用的训练数据的格式为[C, H, W]，因此需要对图像数据重新排列，例如[224, 224, 3]变为[3, 224, 224]；

## 3、构建数据读取器
decode中的代码仅展示了读取一张图片和预处理的方法，但是在实际模型训练时，多采用批量数据读取处理的方式。
因此我们需要采用PaddlePaddle中的[Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html)和[DatasetLoader API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)构建数据读取器。









