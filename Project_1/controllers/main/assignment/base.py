import numpy as np
import pandas as pd


# 宏定义
X = 0 # 四元数下标
Y = 1
Z = 2
W = 3  


########################################## 自定基础函数 ##########################################
# 向量基础函数

# 向量单位化
def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # 返回原始向量
    else:
        return v / norm  # 返回单位化向量

# 向量夹角
def compute_angle(v1, v2):
    # 计算两个向量的夹角（弧度）
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 限制在 [-1, 1] 范围内
    return angle

# 四元数基础函数
def quat_mutiplication(q1, q2):
    # 根据 [x, y, z, w] 的公式
    x = q1[W]*q2[X] + q1[X]*q2[W] + q1[Y]*q2[Z] - q1[Z]*q2[Y]
    y = q1[W]*q2[Y] - q1[X]*q2[Z] + q1[Y]*q2[W] + q1[Z]*q2[X]
    z = q1[W]*q2[Z] + q1[X]*q2[Y] - q1[Y]*q2[X] + q1[Z]*q2[W]
    w = q1[W]*q2[W] - q1[X]*q2[X] - q1[Y]*q2[Y] - q1[Z]*q2[Z]
    return np.array([x, y, z, w])

def quat_rotate(P1, Q):
    Q_prim = np.array([-Q[X], -Q[Y], -Q[Z], Q[W]])
    P2 = quat_mutiplication(quat_mutiplication(Q,P1),Q_prim)
    return P2

def vector_rotate(p1, Q):
    P1 = np.array([p1[X], p1[Y], p1[Z], 0]) # 添加 0
    P2 = quat_rotate(P1, Q)
    return P2[[X,Y,Z]]     # 返回旋转后的向量部分


# 目标中心点基础函数
def compute_target_center(rect, eps=1e-6):

    # 取出四个点
    x0, y0 = rect[0]
    x1, y1 = rect[1]
    x2, y2 = rect[2]
    x3, y3 = rect[3]

    # 计算分母
    denom = (x0 - x2) * (y1 - y3) - (y0 - y2) * (x1 - x3)
    if abs(denom) < eps:
        # 分母接近0，说明两直线平行或共线，无法确定交点
        return None

    # 计算分子中的通项
    det1 = x0 * y2 - y0 * x2
    det2 = x1 * y3 - y1 * x3

    # 计算交点坐标
    x = (det1 * (x1 - x3) - (x0 - x2) * det2) / denom
    y = (det1 * (y1 - y3) - (y0 - y2) * det2) / denom

    center = np.array([x, y])

    return center

# 四边形重新排序函数
def SORT(pts):
    """
    输入：
        pts: numpy 数组，形状 (4,2)，每行是一个 (x,y) 坐标
    返回：
        按 [左上, 左下, 右下, 右上] 排序后的点，形状 (4,2)
    """
    # 1. 按 x 坐标升序，分成左右两组
    pts_sorted = pts[np.argsort(pts[:, 0])]
    left  = pts_sorted[:2]   # x 最小的两个
    right = pts_sorted[2:]   # x 最大的两个

    # 2. 左组按 y 升序：上<下；右组同理
    left  = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    tl, bl = left    # top-left, bottom-left
    tr, br = right   # top-right, bottom-right

    return np.array([tl, bl, br, tr], dtype=pts.dtype)

# 保存数据
def save_data(target_pos_list_buffer, file_name = "target_positions"):
    # 创建一个列表，用于存储所有目标点的字典数据
    rows = []

    for frame_idx, targets in enumerate(target_pos_list_buffer):
        for point_idx, point in enumerate(targets):
            rows.append({
                'frame': frame_idx,
                'point_index': point_idx,
                'x': point[0],
                'y': point[1],
                'z': point[2]
            })

    # 将数据转换为 DataFrame
    df = pd.DataFrame(rows)

    df.to_csv(f'{file_name}.csv', index=False)

    print("保存 CSV 文件成功！")

# 计算2点距离函数
def compute_distance(P1, P2):
    return np.linalg.norm(P1 - P2)

########################################## 自定基础函数 ##########################################
