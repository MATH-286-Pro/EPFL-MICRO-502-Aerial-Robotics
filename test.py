# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def compute_pose_and_normals(target_pos_list, K):
#     """
#     利用图像中正方形的 4 个角点 (逆时针顺序) 和相机内参 K，
#     通过 solvePnP 计算位姿，同时利用两种方法计算平面法向量：
#       1. 通过旋转矩阵 R 得到：normal_R = R * [0, 0, 1]
#       2. 将四个角点转换到相机坐标系后，取两个边向量叉乘计算法向量
#     返回:
#       R: 旋转矩阵（世界到相机）
#       tvec: 平移向量
#       normal_R: 方法1计算得到的法向量
#       normal_pts: 方法2计算得到的法向量（归一化）
#       pts_cam: 四个角点转换到相机坐标系下的坐标（形状为 (4,3)）
#       obj_points: 世界坐标下正方形的顶点（仅用于后续绘图）
#     """
#     # 定义世界坐标系下的正方形顶点（单位正方形, z=0），顺序应与 target_pos_list 对应
#     obj_points = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0],
#         [0.0, 1.0, 0.0]
#     ], dtype=np.float32)
    
#     # 转换图像中的角点坐标，reshape 成 (N,1,2)
#     image_points = np.array(target_pos_list, dtype=np.float32).reshape(-1, 1, 2)
    
#     # 假设无畸变，如有真实标定数据请替换
#     dist_coeffs = np.zeros(5, dtype=np.float32)
    
#     # 调用 solvePnP 求解位姿，注意这里的默认模型是：X_cam = R * X_world + tvec
#     success, rvec, tvec = cv2.solvePnP(
#         objectPoints=obj_points,
#         imagePoints=image_points,
#         cameraMatrix=K,
#         distCoeffs=dist_coeffs,
#         flags=cv2.SOLVEPNP_ITERATIVE
#     )
    
#     if not success:
#         raise ValueError("solvePnP 求解失败，请检查输入数据！")
    
#     # 将旋转向量转换为旋转矩阵 R
#     R, _ = cv2.Rodrigues(rvec)
    
#     # 方法1：利用 R 直接计算法向量
#     # 因为在世界坐标系下正方形位于 z=0 平面，法向量为 [0, 0, 1]
#     normal_R = R @ np.array([0, 0, 1], dtype=np.float32)
    
#     # 将世界坐标中的角点转换到相机坐标系下
#     pts_cam = (R @ obj_points.T).T + tvec.T  # (4,3)
    
#     # 方法2：利用相机坐标系下角点直接计算
#     # 例如，以第1个顶点为基准，取顶点2和顶点4构成两个边向量
#     edge1 = pts_cam[1] - pts_cam[0]
#     edge2 = pts_cam[3] - pts_cam[0]
#     normal_pts = np.cross(edge1, edge2)
#     normal_pts = normal_pts / np.linalg.norm(normal_pts)  # 单位化
    
#     return R, tvec, normal_R, normal_pts, pts_cam, obj_points

# def plot_square_and_normals(pts_cam, normal_R, normal_pts):
#     """
#     在 3D 坐标系中绘制正方形四个角点及两种计算方式获得的法向量（从正方形中心出发）。
#     参数：
#       pts_cam: 正方形角点在相机坐标系下的坐标（(4,3) 数组）
#       normal_R: 方法1 得到的法向量
#       normal_pts: 方法2 得到的法向量
#     """
#     # 计算正方形中心（在相机坐标系下）
#     center = np.mean(pts_cam, axis=0)
    
#     # 箭头长度取正方形任一边长
#     arrow_length = np.linalg.norm(pts_cam[1] - pts_cam[0])
    
#     # 建立 3D 图形
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 绘制正方形边框（闭合多边形）
#     pts_plot = np.vstack((pts_cam, pts_cam[0:1]))
#     ax.plot(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], 'b-o', label='Square corners')
    
#     # 绘制法向量（方法1：利用 R 得到的，红色箭头）
#     ax.quiver(center[0], center[1], center[2],
#               normal_R[0], normal_R[1], normal_R[2],
#               length=arrow_length, color='r', label='Normal from R')
    
#     # 绘制法向量（方法2：利用叉乘角点，绿色箭头）
#     ax.quiver(center[0], center[1], center[2],
#               normal_pts[0], normal_pts[1], normal_pts[2],
#               length=arrow_length, color='g', label='Normal from points')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title("Square Corners & Normal Vectors in Camera Coordinates")
#     ax.legend()
    
#     # 调整视角，便于观察
#     ax.view_init(elev=20, azim=30)
#     plt.show()

# if __name__ == '__main__':
#     # 示例：图像中正方形角点的像素坐标 (逆时针顺序)
#     Target_Pos_list = [
#         (100, 150),  # 顶点1
#         (200, 140),  # 顶点2
#         (210, 250),  # 顶点3
#         (110, 260)   # 顶点4
#     ]
    
#     # 示例相机内参 (fx, fy, cx, cy)，请替换为实际标定参数
#     K = np.array([
#         [800,   0, 320],
#         [  0, 800, 240],
#         [  0,   0,   1]
#     ], dtype=np.float32)
    
#     # 计算位姿和法向量（两种方法）
#     R, tvec, normal_R, normal_pts, pts_cam, obj_points = compute_pose_and_normals(Target_Pos_list, K)
    
#     print("利用 R 得到的法向量：", normal_R)
#     print("利用角点叉乘得到的法向量：", normal_pts)
    
#     # 绘制正方形和法向量
#     plot_square_and_normals(pts_cam, normal_R, normal_pts)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图模块

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

ax.legend()
plt.show()
