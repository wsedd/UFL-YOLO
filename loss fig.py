import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
baseline_df = pd.read_csv('')  # 替换为您的基线模型CSV文件路径
focal_df = pd.read_csv('')  # 替换为您的Focal Loss模型CSV文件路径
nafl_df = pd.read_csv('')  # 替换为您的NAFL模型CSV文件路径

# 提取损失数据
# 假设CSV文件中包含'epoch'和'classification_loss'列
baseline_epochs = baseline_df['                  epoch'].values
baseline_loss = baseline_df['         train/cls_loss'].values

focal_epochs = focal_df['                  epoch'].values
focal_loss = focal_df['         train/cls_loss'].values

nafl_epochs = nafl_df['                  epoch'].values
nafl_loss = nafl_df['         train/cls_loss'].values

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制三条损失曲线
plt.plot(baseline_epochs, baseline_loss, 'b-', linewidth=2, label='Baseline')
plt.plot(focal_epochs, focal_loss, 'g--', linewidth=2, label='With Focal Loss')
plt.plot(nafl_epochs, nafl_loss, 'r-', linewidth=2.5, label='With NAFL')

# 添加标注和美化
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Classification Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# 根据数据范围调整Y轴
max_loss = max(np.max(baseline_loss), np.max(focal_loss), np.max(nafl_loss))
min_loss = min(np.min(baseline_loss), np.min(focal_loss), np.min(nafl_loss))
plt.ylim(min_loss * 0.9, max_loss * 1.1)

plt.tight_layout()

# 保存图像
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
