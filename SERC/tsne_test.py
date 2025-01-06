# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # 假设你有一些数据和标签
# # X 是特征数据，y 是标签（类别），domain 是域（源域或目标域）
# X = np.random.rand(100, 10)  # 示例特征数据
# y = np.random.randint(0, 3, 100)  # 三个类别
# domain = np.random.choice(['source', 'target'], 100)  # 源域和目标域

# # 使用 t-SNE 降维
# tsne = TSNE(n_components=2, random_state=0)
# X_embedded = tsne.fit_transform(X)

# # 创建图形
# plt.figure(figsize=(10, 8))

# # 定义颜色和形状
# colors = {'source': 'blue', 'target': 'orange'}
# markers = {0: 'o', 1: 's', 2: 'D'}  # 分别对应三种类别

# # 绘制每个点
# for i in range(len(X_embedded)):
#     plt.scatter(X_embedded[i, 0], X_embedded[i, 1],
#                 color=colors[domain[i]],
#                 marker=markers[y[i].item()],
#                 alpha=0.7)

# # 添加图例
# plt.legend(handles=[
#     plt.Line2D([0], [0], marker='o', color='w', label='Source - Class 0', 
#                 markerfacecolor=colors['source'], markersize=10),
#     plt.Line2D([0], [0], marker='s', color='w', label='Source - Class 1', 
#                 markerfacecolor=colors['source'], markersize=10),
#     plt.Line2D([0], [0], marker='D', color='w', label='Source - Class 2', 
#                 markerfacecolor=colors['source'], markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='Target - Class 0', 
#                 markerfacecolor=colors['target'], markersize=10),
#     plt.Line2D([0], [0], marker='s', color='w', label='Target - Class 1', 
#                 markerfacecolor=colors['target'], markersize=10),
#     plt.Line2D([0], [0], marker='D', color='w', label='Target - Class 2', 
#                 markerfacecolor=colors['target'], markersize=10),
# ])

# plt.title('t-SNE Visualization of Source and Target Domains')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.grid()
# plt.show()
# plt.savefig("a.jpg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 假设你有一些数据和标签
# X 是特征数据，y 是标签（类别），domain 是域（源域或目标域）
X = np.random.rand(100, 10)  # 示例特征数据
y = np.random.randint(0, 3, 100)  # 三个类别 (0, 1, 2)
domain = np.random.choice(['source', 'target'], 100)  # 源域和目标域

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=0)
X_embedded = tsne.fit_transform(X)

# 获取类别
unique_classes = np.unique(y)

# 为每种类别绘制单独的图
for class_label in unique_classes:
    plt.figure(figsize=(10, 8))
    
    # 筛选当前类别的数据点
    indices = np.where(y == class_label)
    X_class = X_embedded[indices]
    domain_class = domain[indices]
    
    # 定义颜色
    colors = {'source': 'blue', 'target': 'orange'}
    
    # 绘制每个点
    for i in range(len(X_class)):
        plt.scatter(X_class[i, 0], X_class[i, 1],
                    color=colors[domain_class[i]],
                    alpha=0.7)
        
    plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label=f'Source - Class {class_label}', 
                markerfacecolor=colors['source'], markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label=f'Target - Class {class_label}', 
                markerfacecolor=colors['target'], markersize=10),
] ,loc='upper right', fontsize=10)
    # 添加标题和标签
    plt.title(f't-SNE Visualization for Class {class_label}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    
    # 显示图形
    plt.show()
    plt.savefig(f"{class_label}_class.jpg")
    plt.close()