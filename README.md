电力设备类 使用深度学习框架yolov8模型训练变压器漏油数据集 通过训练出的权重，如何进行识别检测电力设备漏油只变压器漏油检测数据集的监测
@[toc]
先进行数据集的标注，在分类到训练集 测试集 ，验证集
对吧。标注oil（也就是油的意思）
变压器漏油识别数据集 变压器漏油检测1分类，oil(漏油)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8e507b4e04e4fb596355adb73ebd44c.png)
一般变压器都用于:

油浸式变压器通常采用油浸自冷式、油浸风冷式和强迫油循环三种冷却方式。采集于油浸式变压器的设备漏油情况，一般用于变电站的无人巡检，代替传统的人工巡检， 共338张。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/38eca9ee2be44dfd865b7fe6d5cdc333.png)
一般数据集划分8：1：1，也可以没有测试集，首先肯定要训练，这个是是主要的。
数据集按9：1分配。训练集304张图片，验证集34张，YOLOtxt标注。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d7c2311675854c0585c3db8bd0533c8e.png)
然后，我们就进行我们的训练，吧
要从官方调用并使用YOLOv5模型进行推理，你需要遵循几个步骤来安装必要的依赖项、加载预训练的模型以及执行推理。以下是详细的步骤：

先调用呗，你可能没有Yolo5
### 1. 安装依赖

首先，确保已经克隆了YOLOv5的仓库，并安装了所有必要的Python依赖项。

```bash
# 克隆YOLOv5仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 安装所需的Python包
pip install -r requirements.txt
```

### 2. 加载预训练模型

YOLOv5有多个版本（如yolov5s, yolov5m, yolov5l, yolov5x），分别代表不同大小和复杂度的模型。你可以根据你的需求选择一个合适的模型加载。

以下是如何加载一个预训练模型的例子：

```python
import torch

# 加载预训练的YOLOv5模型，这里以yolov5s为例
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

如果你想加载自己训练好的模型，可以指定权重文件的路径：

```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/your/best.pt')
```

### 3. 执行推理

加载模型后，你可以对图片或视频执行目标检测任务。下面是一个简单的例子，展示如何对单张图片进行推理：

```python
# 对一张图片执行推理
img = 'https://ultralytics.com/images/zidane.jpg'  # 可替换为你的图片路径
results = model(img)

# 显示结果
results.print()  # 输出结果到控制台
results.show()   # 显示带有边界框的结果图像
```

对于批量处理图片或对整个目录中的图片进行推理，可以这样做：

```python
import glob

# 获取目录中所有图片路径
img_paths = glob.glob('path/to/images/*.jpg')

# 对每张图片执行推理
for img in img_paths:
    results = model(img)
    results.print()
    results.save()  # 将结果保存到runs/detect/exp目录下
```

### 4. 自定义类别处理

如果你训练了一个特定于变压器漏油检测的模型（例如只有一个`oil`类），你可能需要根据模型输出解析特定的信息。可以通过访问`results.pandas().xyxy[0]`获取每个检测对象的详细信息（如坐标、置信度和类别）。

```python
# 假设只对oil类别感兴趣
detections = results.pandas().xyxy[0]  # 转换为pandas DataFrame
oil_detections = detections[detections['name'] == 'oil']
print(oil_detections)
```



训练一个专门用于识别变压器漏油的YOLOv5模型，准备数据、配置模型参数，并进行训练。数据集、编写配置文件以及具体的训练代码。
代码示例，给同学参考，仅供参考。学习。

### 1. 准备工作

确保你已经安装了必要的依赖项：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 2. 数据集组织

假设你的数据集结构如下：

```
transformer_oil_leakage/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- `images/train` 和 `labels/train` 分别包含304张训练图像和它们对应的标注。
- `images/val` 和 `labels/val` 分别包含34张验证图像和它们对应的标注。

### 3. 创建数据集配置文件

创建一个名为 `data.yaml` 的文件来描述你的数据集。内容如下：

```yaml
train: ./transformer_oil_leakage/images/train  # 训练集图片路径
val: ./transformer_oil_leakage/images/val      # 验证集图片路径

nc: 1  # 类别数量，这里是1，因为我们只有一个类别：oil(漏油)
names: ['oil']  # 类别名称列表
```

### 4. 开始训练

在YOLOv5目录下运行以下命令开始训练模型。你可以根据需要调整参数（如`--img`、`--batch-size`等）以适应你的硬件条件。

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

- `--img 640`: 设置输入图像的大小为640x640。可以根据实际情况调整这个尺寸。
- `--batch 16`: 每个批次的图像数量。如果内存允许，可以增加这个数值以加快训练速度。
- `--epochs 100`: 训练周期数。你可以根据需求调整这个值。
- `--data data.yaml`: 指定之前创建的数据集配置文件。
- `--weights yolov5s.pt`: 从YOLOv5的小型版本（yolov5s）开始微调。如果你希望从头开始训练，可以将此参数替换为`--weights ''`。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4e8a7bf7c2f45e08f362d842f12f9a9.png)
### 完整训练脚本示例
训练的时候，路径别搞错了，不然容易报错！

下面是一个完整的训练脚本示例，展示了如何结合上述步骤进行训练：

```python
# 进入YOLOv5目录
import os
os.chdir('path/to/yolov5')  # 替换为你的YOLOv5项目路径

# 执行训练命令
cmd = 'python train.py --img 640 --batch 16 --epochs 100 --data path/to/data.yaml --weights yolov5s.pt'
os.system(cmd)
```


