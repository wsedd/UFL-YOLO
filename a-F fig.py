import matplotlib.pyplot as plt
import pandas as pd

# 创建数据框
data = {
    'Method': ['Faster R-CNN', 'SSD', 'RetinaNet', 'YOLOv5-n', 'YOLOv8-n', 'YOLOv10-n', 'RT-DETR-R18', 'UFL-YOLO (Ours)','Dynamic YOLO','UM-YOLOv10','Rcf-YOLO','EPBC-YOLOv8','UTD-YOLOv5','GCC-Net','Boosting R-CNN'],
    'mAP@0.5': [76.4, 72.3, 76.8, 78.9, 79.6, 81.2, 80.4, 86.7, 85.2,83.8,84.3,84.1,79.4,81.7,79.5],
    'FPS': [23, 58, 36, 165, 152, 215, 57, 196,181,204,221,170,39.2,65,12],
    'Params (M)': [41.5, 24.0, 34.0, 1.9, 3.2, 1.1, 22.7, 1.73,2.9,1.8,1.1,2.6,8.4,5.1,55.4]
}

df = pd.DataFrame(data)

# 创建图表
plt.figure(figsize=(10, 6))
plt.scatter(df['FPS'], df['mAP@0.5'], s=df['Params (M)']*10, alpha=0.6)  # 点大小代表参数量

# 添加标注
for i, row in df.iterrows():
    plt.annotate(row['Method'], (row['FPS'], row['mAP@0.5']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

# 设置坐标轴和标题
plt.xlabel('Frames Per Second (FPS)', fontsize=12)
plt.ylabel('mAP@0.5 (%)', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

# 高亮显示我们的方法
plt.scatter(df.loc[7, 'FPS'], df.loc[7, 'mAP@0.5'], s=df.loc[7, 'Params (M)']*10,
            color='red', edgecolors='black', linewidth=2, label='UFL-YOLO (Ours)')


plt.tight_layout()
plt.show()