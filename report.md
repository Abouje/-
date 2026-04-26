# HW1 实验报告：手写自动微分三层 MLP 完成 EuroSAT 地表覆盖分类

## 1. 实验任务

本实验在不使用 PyTorch、TensorFlow、JAX 自动微分能力的前提下，基于 NumPy 从零实现三层神经网络分类器，并在 EuroSAT_RGB 数据集上完成：

- 数据加载与预处理；
- 模型定义与反向传播；
- 训练、验证、测试；
- 超参数搜索；
- 权重可视化与错例分析。

---

## 2. 数据集与预处理

### 2.1 数据集

- 数据目录：`EuroSAT_RGB/`
- 类别数：10 类
- 图像类型：RGB

### 2.2 输入设置

本次最终实验使用原始分辨率输入，不进行降采样：

- `image_size = 64`
- 输入维度：`64 x 64 x 3 = 12288`

### 2.3 数据划分与标准化

- 训练集/验证集/测试集 = 70% / 15% / 15%（固定随机种子）；
- 标准化使用训练集统计量（mean/std），并应用到验证、测试集。

实现位置：`src/data.py`。

---

## 3. 方法实现

### 3.1 自动微分

在 `src/autograd.py` 中实现动态计算图与自动微分，支持：

- 张量基本算子：`+ - * / @ sum mean reshape`；
- 激活算子：`relu / sigmoid / tanh`；
- 拓扑排序反向传播；
- 广播梯度回收（unbroadcast）。

核心实现：`src/autograd.py`。

### 3.2 三层 MLP

模型定义在 `src/model.py`，最终实验配置为：

`Linear(12288, 1024) -> ReLU -> Linear(1024, 1024) -> ReLU -> Linear(1024, 10)`

### 3.3 训练组件

- 损失函数：交叉熵（`src/losses.py`）；
- 正则化：L2（`src/losses.py`）；
- 优化器：SGD（`src/optim.py`）；
- 学习率衰减：step decay（`src/engine.py`）；
- 最优模型保存：按验证集准确率自动保存 checkpoint。

### 3.4 续训能力

训练脚本支持从 checkpoint 继续训练，参数：`--resume_checkpoint`（`scripts/train.py`）。

---

## 4. 训练配置与执行

最终训练命令：

```bash
python3 scripts/train.py \
  --data_root ./EuroSAT_RGB \
  --output_dir ./results/raw64_h1024_e100 \
  --epochs 100 \
  --batch_size 2048 \
  --image_size 64 \
  --hidden_dim 1024 \
  --activation relu \
  --lr 0.05 \
  --weight_decay 1e-4 \
  --decay_rate 0.9 \
  --decay_every 10
```

对应输出目录：`results/raw64_h1024_e100/`。

---

## 5. 训练过程结果

从 `results/raw64_h1024_e100/history.json` 统计：

- 训练轮数：100
- 最优验证准确率：`0.6667`
- 最优 epoch：`94`
- 最后一轮训练指标：
  - train loss = `0.6748`
  - train acc = `0.8390`
  - val loss = `0.9732`
  - val acc = `0.6454`

### 5.1 Loss 曲线（训练集 + 验证集）

![训练集与验证集 Loss 曲线](./results/raw64_h1024_e100/loss_curve.png)

### 5.2 验证集 Accuracy 曲线

![验证集 Accuracy 曲线](./results/raw64_h1024_e100/val_acc_curve.png)

> 以上两张曲线图来自本次 100 epoch 最终训练的自动保存结果，已作为最终报告图。

---

## 6. 测试集评估

测试命令：

```bash
python3 scripts/test.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/raw64_h1024_e100/best_model.npz \
  --output_dir ./results/raw64_h1024_e100
```

测试结果：

- **Test Accuracy = 0.6652**

混淆矩阵文件：

- 文本：`results/raw64_h1024_e100/confusion_matrix.txt`
- 图像：`results/raw64_h1024_e100/confusion_matrix.png`

---

## 7. 权重可视化与错例分析

分析命令：

```bash
python3 scripts/analysis.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/raw64_h1024_e100/best_model.npz \
  --output_dir ./results/raw64_h1024_e100 \
  --num_error_cases 12
```

输出：

- 第一层权重可视化：`results/raw64_h1024_e100/first_layer_weight_viz.png`

  ![验证集 Accuracy 曲线](./results/raw64_h1024_e100/first_layer_weight_viz.png)

- 错例图：`results/raw64_h1024_e100/error_cases.png`

  ![验证集 Accuracy 曲线](./results/raw64_h1024_e100/error_cases.png)

- 错例文本：`results/raw64_h1024_e100/error_cases.txt`

### 7.1 错分统计与地物特征解释

结合混淆矩阵（类别顺序：AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake），可以看到以下高频错分：

- `PermanentCrop -> AnnualCrop`：81 次；`HerbaceousVegetation -> AnnualCrop`：69 次。农作地类之间颜色分布和块状纹理接近，边界形态也相似。
- `HerbaceousVegetation -> PermanentCrop`：62 次；`HerbaceousVegetation -> Residential`：37 次。草地/灌丛在遥感图中常呈细碎纹理，容易与规则耕作地或低密度建成区混淆。
- `Highway -> River`：52 次；`Highway -> AnnualCrop`：45 次；`Highway -> Residential`：45 次。线性结构在 MLP 展平输入后难以保持拓扑连续性，导致道路与河流、道路与建成区边缘纹理被混淆。
- `River -> Highway`：56 次。河道在局部窗口中常表现为细长带状，与道路的方向性纹理有明显重叠。
- `SeaLake -> Forest`：53 次。部分湖岸样本含大量深色植被背景，蓝绿混合后在全连接特征空间中更靠近森林类。

### 7.2 结合具体错例图片的分析

根据 `results/raw64_h1024_e100/error_cases.txt` 中的实际错分样本，可以给出更具体的地物层面解释：

- `River_1198.jpg (River -> Highway)`：河道宽度较窄、走向近似直线，且两侧地物对比明显，视觉上接近“道路穿越场景”。
- `Highway_2311.jpg (Highway -> AnnualCrop)`：道路占图比例偏小，周围大片农田纹理占主导，模型更容易被背景纹理“拉向”耕地类别。
- `SeaLake_1351.jpg (SeaLake -> Residential)`：样本中水体与岸线建筑/硬化地面同时出现，水体纯度不足，导致模型偏向建成区。
- `PermanentCrop_902.jpg (PermanentCrop -> HerbaceousVegetation)`：规则果园纹理不明显时，与自然植被在颜色与纹理尺度上接近。
- `Residential_1601.jpg (Residential -> Highway)`：密集道路网和建筑屋顶交错，局部窗口内“线状结构”特征过强，诱导模型预测为 Highway。

### 7.3 误差原因归纳

1. **类别间先天相似性**：农作地、草地、果园都具有大面积绿色/黄绿色纹理，类间边界模糊。
2. **线性目标混淆**：道路与河流都具有“细长、方向性强”的纹理特征，且在小区域内形态相近。
3. **场景混合问题**：海湖沿岸、城郊边缘样本常含多类地物，单标签样本会放大决策困难。
4. **模型结构限制**：MLP 对空间邻域关系建模弱，展平输入后不利于保留地物的几何结构。

---

## 8. 结论

在原始 64x64 分辨率输入和 100 epoch 训练设置下，手写自动微分三层 MLP 在 EuroSAT 上达到：

- 最优验证准确率：`0.6667`
- 测试准确率：`0.6652`

结果说明：

1. 在不使用深度学习框架自动求导能力的条件下，手写 MLP 仍能稳定收敛并获得有效分类性能；
2. 保持原始分辨率输入对遥感地物识别有帮助；
3. 误差主要来自视觉特征相近类别，后续可通过更强空间建模结构进一步提升性能。

---

## 9. 复现实验

```bash
# 1) 安装依赖
python3 -m pip install -r requirements.txt

# 2) 100轮原始分辨率训练
python3 scripts/train.py \
  --data_root ./EuroSAT_RGB \
  --output_dir ./results/raw64_h1024_e100 \
  --epochs 100 \
  --batch_size 2048 \
  --image_size 64 \
  --hidden_dim 1024 \
  --activation relu \
  --lr 0.05 \
  --weight_decay 1e-4 \
  --decay_rate 0.9 \
  --decay_every 10

# 3) 测试
python3 scripts/test.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/raw64_h1024_e100/best_model.npz \
  --output_dir ./results/raw64_h1024_e100

# 4) 可视化与错例分析
python3 scripts/analysis.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/raw64_h1024_e100/best_model.npz \
  --output_dir ./results/raw64_h1024_e100
```
