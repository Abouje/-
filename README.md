# EuroSAT 三层 MLP 从零实现（HW1）

本项目按 `hw1.pdf` 要求实现了一个**不依赖 PyTorch/TensorFlow/JAX 自动微分**的三层神经网络分类器，用于 EuroSAT 遥感地表覆盖分类。

## 1. 项目结构

- `src/autograd.py`：Tensor 计算图与自动微分（反向传播）
- `src/model.py`：三层 MLP（`Linear + 激活 + Linear + 激活 + Linear`）
- `src/losses.py`：交叉熵损失与 L2 正则
- `src/optim.py`：SGD（支持 momentum）
- `src/data.py`：数据加载、划分、标准化、mini-batch
- `src/engine.py`：训练/验证/预测与 checkpoint 保存加载
- `src/metrics.py`：准确率与混淆矩阵
- `scripts/train.py`：训练主程序（含学习率衰减、best model 保存、曲线绘制）
- `scripts/search.py`：网格搜索超参数
- `scripts/test.py`：测试集评估 + 混淆矩阵输出
- `scripts/analysis.py`：第一层权重可视化 + 错例可视化
- `report.md`：实验报告模板与实验结论（可直接转 PDF）

## 2. 环境安装

```bash
python3 -m pip install -r requirements.txt
```

## 3. 训练

```bash
python3 scripts/train.py \
  --data_root ./EuroSAT_RGB \
  --output_dir ./results/main_run \
  --hidden_dim 256 \
  --activation relu \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.05 \
  --decay_rate 0.9 \
  --decay_every 5 \
  --weight_decay 1e-4 \
  --image_size 32
```

训练输出：
- `results/main_run/best_model.npz`：验证集最优权重
- `results/main_run/history.json`：训练日志
- `results/main_run/loss_curve.png`：训练/验证 Loss 曲线
- `results/main_run/val_acc_curve.png`：验证集 Accuracy 曲线

## 4. 超参数搜索

```bash
python3 scripts/search.py \
  --data_root ./EuroSAT_RGB \
  --output_csv ./results/grid_search.csv \
  --epochs 10 \
  --lrs 0.05,0.02,0.01 \
  --hidden_dims 128,256,384 \
  --weight_decays 0.0,0.0001,0.001 \
  --activations relu,tanh
```

输出：`results/grid_search.csv`（按验证准确率排序）

## 5. 测试评估

```bash
python3 scripts/test.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/main_run/best_model.npz \
  --output_dir ./results/main_run
```

输出：
- 终端打印测试集 Accuracy
- 终端打印混淆矩阵
- `confusion_matrix.txt`
- `confusion_matrix.png`

## 6. 权重可视化与错例分析

```bash
python3 scripts/analysis.py \
  --data_root ./EuroSAT_RGB \
  --checkpoint ./results/main_run/best_model.npz \
  --output_dir ./results/main_run \
  --num_error_cases 12
```

输出：
- `first_layer_weight_viz.png`：第一层隐藏层权重可视化
- `error_cases.png`：测试集错例图
- `error_cases.txt`：错例文件路径与真值/预测标签
