# Dataset Download Guide

本项目的图片获取采用三级策略：**本地数据集 → Web 爬取 → T2I 生成（最后手段）**。
本文档说明如何下载和配置本地数据集，以及 ImageNet 各版本的区别。

所有数据集统一下载到 `/mnt/hdd/xuran/datasets/` 目录下。

---

## 1. MSCOCO 2017

| 项目 | 值 |
|------|-----|
| 用途 | 基于 caption 的文本搜索（5 条 caption/图片） |
| 训练图片 | 118,287 张 |
| 大小 | ~19 GB |
| 分辨率 | 多数 640×480 或更高 |

### 下载命令

```bash
mkdir -p /mnt/hdd/xuran/datasets/coco && cd /mnt/hdd/xuran/datasets/coco

# 训练图片 (~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip

# 标注文件 (~241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

### 目录结构

```
/mnt/hdd/xuran/datasets/coco/
├── train2017/                          # 118,287 张图片
└── annotations/
    └── captions_train2017.json         # caption 标注
```

### 启用

```yaml
# config/pipeline.yaml
local_datasets:
  mscoco:
    enabled: true
```

---

## 2. Open Images V7

| 项目 | 值 |
|------|-----|
| 用途 | 基于 class label 的文本搜索 |
| 完整训练集 | ~9M 张图片（~570 GB） |
| 建议 | 只下载需要的类别子集 |

### 下载命令

```bash
mkdir -p /mnt/hdd/xuran/datasets/open_images && cd /mnt/hdd/xuran/datasets/open_images

# 1) 类别描述文件 (~1MB)
wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv

# 2) Human-verified image-level labels (~450MB)
wget https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv

# 3) 下载图片（选择以下任一方式）
```

**方式 A：使用官方工具按类别下载（推荐）**

```bash
pip install openimages

# 下载特定类别，每个类别最多 5000 张
python -m openimages.download --base_dir /mnt/hdd/xuran/datasets/open_images/train \
    --labels "Knife" "Baseball bat" "Camera" "Kitchen" "Bridge" "Bottle" \
    "Car" "Person" "Dog" "Cat" "Chair" "Table" \
    --format pascal --csv_dir /mnt/hdd/xuran/datasets/open_images --limit 5000
```

**方式 B：使用 AWS CLI 下载完整训练集**

```bash
pip install awscli
aws s3 --no-sign-request sync \
    s3://open-images-dataset/train \
    /mnt/hdd/xuran/datasets/open_images/train
```

### 目录结构

```
/mnt/hdd/xuran/datasets/open_images/
├── train/                                              # 图片文件 (xxxxx.jpg)
├── oidv7-class-descriptions.csv                        # 类别 ID → 名称映射
└── oidv7-train-annotations-human-imagelabels.csv       # 图片 → 类别标注
```

### 启用

```yaml
# config/pipeline.yaml
local_datasets:
  open_images:
    enabled: true
```

---

## 3. ImageNet (ILSVRC 2012)

| 项目 | 值 |
|------|-----|
| 用途 | 基于 synset 描述的文本搜索 |
| 训练图片 | 1,281,167 张 |
| 类别 | 1,000 个 synset |
| 大小 | ~138 GB（训练集） |

### 下载方式

ImageNet 需要注册后才能下载。

**方式 A：官网下载**

1. 注册 https://image-net.org/ 并等待审核通过
2. 登录后访问 https://image-net.org/download-images.php
3. 下载 **ILSVRC2012_img_train.tar** (~138GB)
4. 下载 synset 映射文件

**方式 B：Kaggle 下载（推荐，不需要额外审核）**

```bash
pip install kaggle
# 需要先配置 ~/.kaggle/kaggle.json
kaggle competitions download -c imagenet-object-localization-challenge \
    -p /mnt/hdd/xuran/datasets/imagenet/
```

**方式 C：Hugging Face 下载**

```bash
# 需要先 huggingface-cli login
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('ILSVRC/imagenet-1k', split='train', cache_dir='/mnt/hdd/xuran/datasets/imagenet_hf')
"
```

### 解压（官网/Kaggle tar 格式）

```bash
cd /mnt/hdd/xuran/datasets/imagenet
mkdir -p ILSVRC/Data/CLS-LOC/train

# 解压主 tar
tar -xf ILSVRC2012_img_train.tar -C ILSVRC/Data/CLS-LOC/train/

# 解压每个 synset 子 tar
cd ILSVRC/Data/CLS-LOC/train
for f in *.tar; do
    d="${f%.tar}"
    mkdir -p "$d"
    tar -xf "$f" -C "$d"
    rm "$f"
done
```

### 获取 synset 映射文件

```bash
cd /mnt/hdd/xuran/datasets/imagenet

# 从 Kaggle 数据中提取，或手动下载
# 文件格式：每行 "nXXXXXXXX synset_name, synonym1, synonym2, ..."
# 例如：n01440764 tench, Tinca tinca
```

### 目录结构

```
/mnt/hdd/xuran/datasets/imagenet/
├── ILSVRC/Data/CLS-LOC/train/
│   ├── n01440764/          # 每个 synset 一个子目录
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...                 # ~1,000 个 synset
└── LOC_synset_mapping.txt  # synset ID → 描述映射
```

### 启用

```yaml
# config/pipeline.yaml
local_datasets:
  imagenet:
    enabled: true
```

---

## 4. CIFAR — 不适用

CIFAR-10/100 图片分辨率仅 **32×32**，远低于本项目要求的 **512×512** 最小尺寸，不适合使用。

---

## 5. 启用后的完整配置

下载完所有需要的数据集后，在 `config/pipeline.yaml` 中更新：

```yaml
local_datasets:
  enabled: true
  root: "/mnt/hdd/xuran/datasets"
  mscoco:
    enabled: true       # 下载完成后改为 true
    images_dir: "coco/train2017"
    annotations: "coco/annotations/captions_train2017.json"
  open_images:
    enabled: true       # 下载完成后改为 true
    images_dir: "open_images/train"
    labels_csv: "open_images/oidv7-class-descriptions.csv"
    annotations_csv: "open_images/oidv7-train-annotations-human-imagelabels.csv"
  imagenet:
    enabled: true       # 下载完成后改为 true
    images_dir: "imagenet/ILSVRC/Data/CLS-LOC/train"
    synsets_file: "imagenet/LOC_synset_mapping.txt"
```

启用后，Path 1/5/6 获取图片时会自动按此顺序搜索：
1. **本地数据集**（MSCOCO → Open Images → ImageNet）
2. **Web 爬取**（Wikimedia / Openverse / Pexels / Pixabay）
3. **T2I 生成**（SD 3.5 Large Turbo）— 仅在前两步均失败时使用

---

## 6. 磁盘空间估算

| 数据集 | 大小 | 图片数 | 搜索方式 |
|--------|------|--------|----------|
| MSCOCO | ~19 GB | 118K | Caption 文本匹配 |
| Open Images（子集） | ~50-100 GB | 按需 | Class label 匹配 |
| ImageNet | ~138 GB | 1.28M | Synset 描述匹配 |

**建议优先级：MSCOCO（最小、caption 质量最高）> ImageNet（覆盖面广）> Open Images（按需下载子集）**

---

## 附录：ImageNet 各版本对比

ImageNet 项目从 2009 年启动至今，产生了多个数据集版本。以下是完整对比。

### A. 数据集版本

| 版本 | 图片数 | 类别数 | 大小 | 说明 |
|------|--------|--------|------|------|
| **ImageNet Full** | 14,197,122 | 21,841 synsets | ~1.3 TB | 完整数据集，层级结构（9 级） |
| **ImageNet-21K** | 14,197,122 | 21,841 | ~1.3 TB | = Full ImageNet，含父类节点（如 "mammal"） |
| **ImageNet-21K-P** | 12,358,688 | 11,221 | ~250 GB | 2021 年清洗版，删除了一半类别但仅减少 13% 图片 |
| **ImageNet-1K (ILSVRC)** | 1,431,167 | 1,000 | ~138 GB | 最常用版本，仅含叶节点类别（如 "German shepherd"） |
| **Tiny ImageNet** | 100,000 | 200 | ~237 MB | 教学用，64×64 分辨率 |
| **ImageNet-V2** | 30,000 | 1,000 | - | 2019 年按相同方法新建的测试集 |
| **ImageNet-C** | - | 1,000 | - | 2019 年，加入各种 corruption 的鲁棒性测试集 |

> **本项目使用 ImageNet-1K (ILSVRC 2012)**，这是学术界使用最广泛的版本。

### B. ILSVRC 竞赛历年变化 (2010-2017)

| 年份 | 数据集变化 | 新增任务 | 冠军方法 | Top-5 错误率 |
|------|-----------|----------|---------|-------------|
| **2010** | 首次举办，1000 类 | 图像分类 | Linear SVM + HoG/LBP | 28.2% |
| **2011** | 替换了 321 个 synset（去除难以定位的类别如 "New Zealand beach"） | 目标定位 | XRCE: SVM + Fisher Vectors | 25.8% |
| **2012** | 替换 90 个 synset 为狗的品种（细粒度分类）。**此后类别不再变化** | - | **AlexNet**（CNN 突破） | 15.3% |
| **2013** | 数据集不变 | 目标检测 | Clarifai (分类), OverFeat (定位) | 11.7% |
| **2014** | 检测数据量大幅增加 | - | **GoogLeNet** (分类), **VGGNet** (定位) | 6.7% |
| **2015** | 数据集不变 | - | **ResNet**（超越人类水平） | 3.6% |
| **2016** | 数据集不变 | - | CUImage (6 模型 ensemble) | 2.99% |
| **2017** | 数据集不变，**最后一届** | - | **SENet** (Squeeze-and-Excitation) | 2.25% |

**关键结论：**

1. **2012 年之后数据集完全相同**：ILSVRC 2012-2017 使用的训练集、验证集完全一致，只是每年参赛方法不同
2. **下载任何一年的版本都等价**：`ILSVRC2012_img_train.tar` = `ILSVRC2014_img_train.tar` 等
3. **为什么有不同年份的名字**：因为竞赛每年举办，但底层数据从 2012 起就固定了
4. **推荐下载 ILSVRC 2012**：这是最广泛使用和引用的版本标识

### C. ImageNet-1K vs ImageNet-21K

| 特征 | ImageNet-1K | ImageNet-21K |
|------|-------------|--------------|
| 类别数 | 1,000 | 21,841 |
| 图片数 | 1.28M | 14.2M |
| 大小 | ~138 GB | ~1.3 TB |
| 类别结构 | 仅叶节点（如 "Golden Retriever"） | 含层级父节点（如 "mammal" → "dog" → "Golden Retriever"） |
| 官方划分 | 有 train/val/test | 无官方划分 |
| 标注质量 | ~94% 准确 | 较低，需清洗 |
| 典型用途 | 分类基准、特征提取 | 大规模预训练 |
| 本项目是否需要 | **是（推荐）** | 否（太大，收益有限） |

### D. 数据质量说明

- ImageNet-1K 验证集约 **6% 标签错误**，训练集约 **10% 有歧义或错误标签**
- 2021 年 ImageNet 团队对非人物类别中的人脸进行了模糊处理（影响 17% 图片）
- 2021 年移除了 2,702 个 "person subtree" 类别以减少偏见
