
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图模块

# # 读取 CSV 文件
# df = pd.read_csv('target_positions.csv')

# # 筛选 point_index 为 4 的数据，并按 frame 排序
# df_center = df[df['point_index'] == 4].sort_values('frame').reset_index(drop=True)
# num_points = len(df_center)

# # 构造颜色列表：颜色从蓝色到黑色（RGB: (0,0,1) 到 (0,0,0)）
# colors = [(0, 0, 1 - idx/(num_points - 1)) for idx in range(num_points)]

# # 创建 3D 图形
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制连线表示轨迹（灰色虚线）
# ax.plot(df_center['x'], df_center['y'], df_center['z'], linestyle='--', color='gray', alpha=0.5)

# # 绘制所有点（采样点），使用渐变颜色
# for idx, row in df_center.iterrows():
#     color = colors[idx]
#     ax.scatter(row['x'], row['y'], row['z'], color=color, s=100)

# # 标注起点（第一个点，用红色圆圈标注）
# start = df_center.iloc[0]
# ax.scatter(start['x'], start['y'], start['z'], s=200, facecolors='none',
#            edgecolors='red', marker='o', linewidths=3, label='Start')

# # 标注终点（最后一个点，用红色叉号标注）
# end = df_center.iloc[-1]
# ax.scatter(end['x'], end['y'], end['z'], s=200, color='red', marker='X', linewidths=2, label='End')

# # 用粉色标注预设的五个点
# # 给定的五个点（X, Y, Z 坐标）
# pink_points = np.array([
#     [2.12, 1.84, 1.24],
#     [5.12, 2.30, 0.78],
#     [7.20, 3.27, 1.29],
#     [5.30, 6.74, 1.19],
#     [2.52, 5.50, 1.04]
# ])
# # 绘制粉色点，并加上标签
# ax.scatter(pink_points[:, 0], pink_points[:, 1], pink_points[:, 2], 
#            color='pink', marker='o', s=150, label='Gate')

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.legend()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图模块




def set_axes_equal(ax):
    '''
    为了 3D 图等比例显示，手动把 x,y,z 的 lim 调成相同范围
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim3d(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim3d(z_mid - max_range/2, z_mid + max_range/2)





# 读取 CSV 文件
df = pd.read_csv('target_positions.csv')

# 筛选 point_index 为 4 的数据，并按 frame 排序
df_center = df[df['point_index'] == 4].sort_values('frame').reset_index(drop=True)
num_points = len(df_center)

# 构造颜色列表：颜色从蓝色到黑色（RGB: (0,0,1) 到 (0,0,0)）
colors = [(0, 0, 1 - idx/(num_points - 1)) for idx in range(num_points)]

# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制连线表示轨迹（灰色虚线）
ax.plot(df_center['x'], df_center['y'], df_center['z'], linestyle='--', color='gray', alpha=0.5)

# 绘制所有点（采样点），使用渐变颜色
for idx, row in df_center.iterrows():
    color = colors[idx]
    ax.scatter(row['x'], row['y'], row['z'], color=color, s=100)

# 标注起点（第一个点，用红色圆圈标注）
start = df_center.iloc[0]
ax.scatter(start['x'], start['y'], start['z'], s=200, facecolors='none',
           edgecolors='red', marker='o', linewidths=3, label='Start')

# 标注终点（最后一个点，用红色叉号标注）
end = df_center.iloc[-1]
ax.scatter(end['x'], end['y'], end['z'], s=200, color='red', marker='X', linewidths=2, label='End')

# 用粉色标注预设的五个点
# 给定的五个点（X, Y, Z 坐标）
pink_points = np.array([
    [2.12, 1.84, 1.24],
    [5.12, 2.30, 0.78],
    [7.20, 3.27, 1.29],
    [5.30, 6.74, 1.19],
    [2.52, 5.50, 1.04]
])
# 绘制粉色点，并加上标签
ax.scatter(pink_points[:, 0], pink_points[:, 1], pink_points[:, 2], 
           color='pink', marker='o', s=150, label='Gate')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

set_axes_equal(ax)


ax.legend()
plt.show()
