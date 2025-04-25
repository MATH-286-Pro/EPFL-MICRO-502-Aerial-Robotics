import numpy as np
import time
import cv2

import pandas as pd


# from assignment.filter import *
# from assignment.planning import *

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:

# sensor_data 字典数据内容：
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position

# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity

# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)

# "roll":  Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw":   Yaw angle (rad)

# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# sensor_data 其他暂时用不到的数据
# "t":
# "range_front"
# "range_left"
# "range_back"
# "range_right"
# "range_down"
# "rate_roll" ...



# A link to further information on how to access the sensor data 
# on the Crazyflie hardware for the hardware practical can be found here: 
# https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate



# 宏定义
X = 0 # 四元数下标
Y = 1
Z = 2
W = 3  
arc2deg = 180/np.pi
deg2arc = np.pi/180

GATE = np.array([
    [2.12, 1.84, 1.24],
    [5.12, 2.30, 0.78],
    [7.20, 3.27, 1.29],
    [5.30, 6.74, 1.19],
    [2.52, 5.50, 1.04]])


# 用户定义全局变量
Drone_Controller = None
Total_Time       = 0
Draw             = False # 是否绘制过轨迹
Explore_State    = 0     # 0 代表在探索中，1 代表探索完毕

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




# 定义无人机类
class Class_Drone_Controller:

    def __init__(self, sensor_data, camera_data):

        # 基本参数
        self.f_pixel = 161.013922282   # 相机焦距
        self.vector_Drone2Cam_DroneFrame = np.array([0.03,0.00,0.01]) # 无人机中心到相机偏移向量
        self.camera_size = [300,300]

        self.cam_center_x = self.camera_size[X] / 2 # 像素中心点 x
        self.cam_center_y = self.camera_size[Y] / 2 # 像素中心点 y

        self.points_filter_threshold = 0.5 # 目标点过滤阈值

        # 无人机信息
        self.sensor_data     = None  # 无人机传感器数据
        self.camera_data     = None  # 相机数据
        self.camera_data_BGR = None  # 相机数据 BGR

        # 实时数据 (大写代表实时数据)
        self.Drone_POS_GLOBAL      = None  
        self.Camera_POS_GLOBAL     = None

        self.IMAGE_POINTS_2D       = None  # 2D 图像方框列表
        self.IMAGE_TARGET_VEC      = None  # 3D 目标方向
        self.IMAGE_TARGET_VEC_list = []    # 3D 目标方框列表，0-3为矩形的四个点，4为中心点

        self.YAW_TARGET    = None
        self.YAW_NORMAL    = None  

        # 缓存数据 (三角定位)
        self.Drone_Pos_Buffer             = [] # 位置缓存
        self.Drone_Target_Vec_Buffer      = [] # 方向缓存
        self.Drone_Target_Vec_List_Buffer = [] # 方向列表缓存
        self.min_cumulative_baseline      = 0.5  # 设定累计基线距离阈值 #00FF00

        self.target_pos_list_buffer = [] # 目标点 4+1 列表
        self.target_pos_list_Valid  = [] # 目标点 4+1 列表    [数据处理后]

        # 路径数据记录
        self.RACING_POINTS_COMMAND = []

        self.AT_GATE             = False # 是否到达 Gate
        self.EXAMING             = True
        self.RACING_EXPLORE      = 0     

        self.RACING_POINT_INDEX  = [] # 记录索引，用于记录 某个Gate 起始 index

        # 视觉命令锁
        self.LOCK = False
        self.IMG_LAST_DIRECTION = None # 上一个视觉方向

        # 巡航
        self.RACING_INDEX = 0
        self.RACING_PATH  = None

        self.timer = None # 基于时间的参数
        self.racing_path  = None 
        self.racing_time  = None

        # 起飞状态
        self.takeoff = False

        # 启动函数
        self.update(sensor_data, camera_data)  # 更新数据
        self.Record_Start_Point_command()      # 记录起始位置
        self.Generate_Scan_Sequence()          # 生成扫描偏移序列

    ########################################## 更新函数 ##########################################

    #00FF00 更新无人机位置 + 更新相机数据 #00FF00
    def update(self, sensor_data, camera_data):
        
        # 更新数据 + 相机
        self.sensor_data  = sensor_data
        self.camera_data  = camera_data

        # 原始图像转为 BGR 图像（忽略 Alpha 图层）
        b, g, r, a           = cv2.split(self.camera_data)
        bgr_image            = cv2.merge([b, g, r])
        self.camera_data_BGR = bgr_image

        # 更新位置 + 相机目标
        self.update_drone_quat()                        # 更新无人机四元数
        self.update_drone_position_global()             # 更新无人机坐标
        self.update_camera_position_global()            # 更新相机坐标
        self.update_IMAGE_TO_VEC_LIST(DEBUG = True)     # 相机坐标系下目标位置列表


        # 更新 三角定位 4+1 列表
        self.update_Target_List_with_Buffer()               # 更新目标点列表 [slef.target_pos_list_buffer] 列表数据
        #  self.update_Target_list_Filtered_CallBack()      # 数据滤波
        #    self.check_target_switch() # 检测目标切换       # 是否切换目标

        # 检测
        self.check_target_AtGate()                       # 检测目标点是否到达
        # self.check_target_switch()                       # 检测目标切换
        self.check_is_near_gate_Vision(DEBUG = False)


        # 更新 YAW 角度
        self.Compute_YAW_TARGET() # [依赖 update_IMAGE_TO_VEC_LIST] 
        self.Compute_YAW_NORMAL() # [依赖 update_Target_List_with_Buffer]

    ########################################## 传感器函数 ##########################################
    def update_drone_quat(self):
        quat = np.array([self.sensor_data['q_x'], 
                         self.sensor_data['q_y'], 
                         self.sensor_data['q_z'], 
                         self.sensor_data['q_w']])
        self.Q = quat.copy()  # 复制四元数

    def update_drone_position_global(self):
        position = np.array([self.sensor_data['x_global'], 
                             self.sensor_data['y_global'], 
                             self.sensor_data['z_global']])
        self.Drone_POS_GLOBAL = position
        return position

    def update_camera_position_global(self):
        P_Drone_global = self.Drone_POS_GLOBAL      # 无人机全局坐标系下位置
        Q_Drone2World  = self.Q                     # 无人机四元数
        P_Drone2Cam_Shift_global = vector_rotate(self.vector_Drone2Cam_DroneFrame, Q_Drone2World)  # 无人机坐标系下相机位置
        P_Cam_global   = P_Drone_global + P_Drone2Cam_Shift_global     
        self.Camera_POS_GLOBAL =  P_Cam_global   # 相机全局坐标系下位置
    
    ########################################## 状态检测函数 ##########################################
    
    # 基于距离检测
    def check_target_AtGate(self):
        
        if self.AT_GATE:
            return True
        
        else:
            try:
                target_pos = self.target_pos_list_Valid[-1][4]   # 目标位置
                drone_pos  = self.update_drone_position_global() # 无人机位置

                dist = compute_distance(target_pos, drone_pos)   # 计算距离

                # 到达目标点范围
                if dist <= 0.5:  #00FF00 后续需要调整

                    # 第一次检测到到达范围
                    self.AT_GATE = True   

                    print("到达目标点范围！")

                    return True
                
                else:
                    self.AT_GATE = False # 重新开始
                    return False
                
            except IndexError:
                return False
    
    # 基于检测位置突变
    def check_target_switch(self):
        
        if len(self.target_pos_list_Valid) == 0:
            return False
        
        if len(self.target_pos_list_Valid) != 0:

            # 检测从 0-> 1 的突变
            if len(self.target_pos_list_Valid) == 1:
                self.RACING_EXPLORE += 1
                self.RACING_POINT_INDEX.append(0) # 记录索引
                print(self.RACING_EXPLORE-1, "->", self.RACING_EXPLORE, "目标点切换！")  

                self.AT_GATE = False # 重新开始

                return True

            # 检测到下一个点
            if len(self.target_pos_list_Valid) >= 2:
                prev = self.target_pos_list_Valid[-2][4]  # 目标位置
                curr = self.target_pos_list_Valid[-1][4]  # 目标位置
                delta = compute_distance(prev, curr) # 计算目标点差值

                if delta >= 2.0: 
                    self.RACING_EXPLORE += 1
                    self.RACING_POINT_INDEX.append(len(self.target_pos_list_Valid) - 1)
                    print(self.RACING_EXPLORE-1, "->", self.RACING_EXPLORE, "目标点切换！") 

                    self.AT_GATE = False # 重新开始

                    return True
                else:
                    return False        


    # 基于图像识别
    # 检测粉色是否出现在边缘
    def check_is_near_gate_Vision(self, DEBUG = False):
        """
        检测输入的二维点集合（如 (N,2) 的 numpy 数组）中是否至少有两个点的 X 或 Y 坐标等于 0 或 300。
        """
        if self.IMAGE_POINTS_2D is not None:
            pts = self.IMAGE_POINTS_2D[0:4] # 取出四个点

            count = 0
            for pt in pts:
                x, y = pt
                if x < 2 or x > 298 or y < 2 or y > 298:
                    count += 1
            if count >= 2:
                if DEBUG:
                    print("检测到方形超出边框")
                return True
            else:
                return False
        else:
            return False
    ########################################## 坐标变换函数 ##########################################
    def Convert_Frame_Drone2World(self, P_DroneFrame):
        Q_Drone2World = self.Q  
        P_WorldFrame  = vector_rotate(P_DroneFrame, Q_Drone2World)    # Body Frame -> World Frame
        return P_WorldFrame

    def Convert_Frame_Cam2Drone(self, P_CamFrame):
        cam_delta_x = P_CamFrame[0] - self.cam_center_x
        cam_delta_y = P_CamFrame[1] - self.cam_center_y
        vector_DroneFrame = np.array([self.f_pixel, -cam_delta_x, -cam_delta_y])
        return vector_DroneFrame

    def Convert_Frame_CamNormal(self, P_CamFrame):
        cam_delta_x = P_CamFrame[0] - self.cam_center_x
        cam_delta_y = P_CamFrame[1] - self.cam_center_y
        vector_DroneFrame = np.array([cam_delta_x, -cam_delta_y])
        return vector_DroneFrame
    ########################################## 图像处理函数 ##########################################
    # 图像 -> 粉色 mask
    def img_BGR_to_PINK(self, DEBUG = False):

        bgr = self.camera_data_BGR.copy()

        # OpenCV 是 BGR 顺序！
        upper_pink  = np.array([255, 185, 255])  # B, G, R
        lower_pink  = np.array([190, 55, 190])   
        binary_mask = cv2.inRange(bgr, lower_pink, upper_pink)

        # 可视化 mask 和提取结果
        if DEBUG:
            pink_only = cv2.bitwise_and(bgr, bgr, mask=binary_mask)      # 应用 mask 扣图
            cv2.imshow("Pink", pink_only)                                # 粉色图

        return binary_mask

    # 图像 -> 特征点
    def update_IMAGE_TO_POINTS_2D(self, binary_mask, DEBUG = False):
        
        # 1.轮廓提取
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2.轮廓近似 (遍历所有轮廓，找到拟合为四边形且面积最大的那个)
        largest_rect = None
        max_area = 0  # 用于记录当前最大的面积
        for cnt in contours:
            # 计算轮廓周长
            peri = cv2.arcLength(cnt, True)

            # 多边形拟合，epsilon 控制拟合精度，通常是周长的 1-5%
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 判断是否是四边形
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    largest_rect = approx
            
        # 复制帧，防止闪烁
        Feature_Frame = self.camera_data_BGR.copy()

        # 特征点提取
        if largest_rect is not None:
            largest_rect     = np.squeeze(largest_rect, axis=1)                     # 将 4x1x2 的数组转换为 4x2 的数组
            rect_center      = compute_target_center(largest_rect)                  # np.Float64
            largest_rect     = SORT(largest_rect)                                   # 点排序
            target_rect      = np.append(largest_rect, [rect_center], axis = 0)     # 添加中心点

            # 更新图像点
            self.IMAGE_POINTS_2D = target_rect
        
        else:
            self.IMAGE_POINTS_2D = None # 没有找到目标

        # 是否画图
        if DEBUG and self.IMAGE_POINTS_2D is not None:
            length = len(self.IMAGE_POINTS_2D)
            increment = int(255/(length+1))  # 计算增量
            green_value = increment
            for x, y in self.IMAGE_POINTS_2D:
                cv2.circle(Feature_Frame, (int(x), int(y)), 5, (0, green_value, 0), -1)
                green_value += increment
            cv2.imshow("Rectangle Corners", Feature_Frame)
        else:
            cv2.imshow("Rectangle Corners", Feature_Frame)



    # 图像 -> 方向向量列表
    def update_IMAGE_TO_VEC_LIST(self, DEBUG = False):

        # 初始化
        Vector_Cam2Target_WorldFrame_list = []

        # 图像处理
        cv2.waitKey(1) # 如果放在 return 后面会报错
        binary_mask = self.img_BGR_to_PINK(DEBUG)            # 抠图
        self.update_IMAGE_TO_POINTS_2D(binary_mask, DEBUG)   # 更新图像2D特征点
        
        # 计算向量
        if self.IMAGE_POINTS_2D is not None:
            
            for cam_point in self.IMAGE_POINTS_2D:
                # 目标方向：相机坐标系 -> 无人机坐标系
                Vector_Cam2Target_DroneFrame = self.Convert_Frame_Cam2Drone(cam_point)    
                Vector_Cam2Target_DroneFrame = unit_vector(Vector_Cam2Target_DroneFrame)  

                # 目标方向：无人机坐标系 -> 世界坐标系
                Vector_Cam2Target_WorldFrame = self.Convert_Frame_Drone2World(Vector_Cam2Target_DroneFrame)
                Vector_Cam2Target_WorldFrame = unit_vector(Vector_Cam2Target_WorldFrame)  

                Vector_Cam2Target_WorldFrame_list.append(Vector_Cam2Target_WorldFrame)    # 添加到列表中

            self.IMAGE_TARGET_VEC_list = Vector_Cam2Target_WorldFrame_list 

        else :
            self.IMAGE_TARGET_VEC_list = None


    ########################################## 三角定位部分 ######################################################
    def triangular_positioning(self,
                           P_WorldFrame_New, 
                           P_WorldFrame_Old,
                           Vector_Direct_Cam2Target_WorldFrame_New,
                           Vector_Direct_Cam2Target_WorldFrame_Old): 

        # 如果方向向量过于接近，返回 None
        if np.array_equal(Vector_Direct_Cam2Target_WorldFrame_New, Vector_Direct_Cam2Target_WorldFrame_Old):
            return None
        else:
            # 重命名
            r0 = Vector_Direct_Cam2Target_WorldFrame_Old
            r1 = Vector_Direct_Cam2Target_WorldFrame_New
            P0 = P_WorldFrame_Old
            P1 = P_WorldFrame_New

            # 求解线性方程组
            A = np.array([[r0 @ r0, -r0 @ r1],
                          [r0 @ r1, -r1 @ r1]])
            b = np.array([[r0 @ (P1 - P0)],
                          [r1 @ (P1 - P0)]])
            
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None) # 最小二乘法求解，更稳定
            except np.linalg.LinAlgError:
                print("Error: Singular matrix, cannot solve the system of equations.")
                return None

            # 计算目标位置
            T0 = P0 + x[0] * r0
            T1 = P1 + x[1] * r1
            T = (T0 + T1) / 2

        return T

    ############################################# 三角定位，缓存更新 ############################################# 
    def update_Target_List_with_Buffer(self):

        # 如果视野内无目标，不能将 None 添加到缓存中
        # #0000FF 大写代表实时更新数据，实时数据会包含 None #0000FF
        if (self.Drone_POS_GLOBAL is not None) and (self.IMAGE_TARGET_VEC_list is not None) and (not self.check_is_near_gate_Vision()):

            # 同时更新 Buffer
            self.Drone_Pos_Buffer.append(self.Drone_POS_GLOBAL)
            self.Drone_Target_Vec_List_Buffer.append(self.IMAGE_TARGET_VEC_list)
        
        # 至少两帧数据
        if len(self.Drone_Pos_Buffer) >= 2: 

            # 计算 位移 + 角度变化
            dist_difference  = np.linalg.norm(self.Drone_Pos_Buffer[-1] - self.Drone_Pos_Buffer[0])
            angle_differnece = compute_angle(self.Drone_Target_Vec_List_Buffer[-1][4], self.Drone_Target_Vec_List_Buffer[0][4])

            # 移动距离大于 最小设定值
            # 1.更新 List_Buffer
            # 2.回调函数滤波 List_Filtered
            if dist_difference >= self.min_cumulative_baseline and angle_differnece >= 0.01:
                
                # 初始化
                Target_Pos_list = np.zeros((5,3)) # 5 个目标位置

                # 计算 5 个目标位置
                for i in range(5):
                    Target_Pos_list[i] = self.triangular_positioning(self.Drone_Pos_Buffer[-1], 
                                                                 self.Drone_Pos_Buffer[0],
                                                                 self.Drone_Target_Vec_List_Buffer[-1][i],
                                                                 self.Drone_Target_Vec_List_Buffer[0][i])

                # 缓存清空 #FF0000 需要修改逻辑，充分利用数据
                self.Drone_Pos_Buffer             = [] # 清空缓存
                self.Drone_Target_Vec_List_Buffer = []

                # 更新目标值
                self.target_pos_list_buffer.append(Target_Pos_list) # 目标缓存

                # 回调函数滤波
                self.update_Target_list_Filtered_CallBack() # 目标点数据处理函数




    # Filter 函数 (三角定位 目标点数据 处理函数) 👆
    # 作为回调函数，在每次 self.target_pos_list_buffer 更新时被调用
    # 否则如果 self.target_pos_list_buffer 不更新，并且最后两个数据接近，则会一直添加到 self.target_pos_list_Valid 中
    def update_Target_list_Filtered_CallBack(self):
        if len(self.target_pos_list_buffer) >= 2:
            P_new = self.target_pos_list_buffer[-1][4] # 最新目标点
            P_old = self.target_pos_list_buffer[-2][4] # 上一个目标点
            P_Diff = compute_distance(P_new, P_old)    # 计算变化量

            # 过滤目标点
            if P_Diff <= 0.2:
                self.target_pos_list_Valid.append(self.target_pos_list_buffer[-1]) # 目标点 4+1 缓存

                self.check_target_switch() # 检测目标切换

                #FF0000 测试 打印中心点
                # print(self.target_pos_list_Valid[-1][4])

    ############################################# 计算目标 YAW #############################################
    def Compute_YAW_TARGET(self):
        if self.IMAGE_TARGET_VEC_list is not None:
            Vector_3D = self.IMAGE_TARGET_VEC_list[4]
            self.YAW_TARGET = np.arctan2(Vector_3D[1], Vector_3D[0])  # 计算 YAW 角度
        else: 
            return None
    
    def Compute_YAW_NORMAL(self):

        # points_global = self.target_pos_list_buffer[-1]

        if len(self.target_pos_list_buffer) > 0:
            # points_global = np.mean(self.target_pos_list_buffer, axis=0) # 计算平均值
            points_global = self.target_pos_list_buffer[-1]

            vec1 = points_global[0] - points_global[3]
            vec2 = points_global[1] - points_global[2]
            vec = 0.5*(vec1 + vec2) 
            theta = np.arctan2(vec[1], vec[0])  # 计算 YAW 角度

            normal_angle = theta - np.pi / 2
            normal_vector = np.array([np.cos(normal_angle), np.sin(normal_angle), 0]) #FF0000 注意这个是不精确的法向量

            self.YAW_NORMAL = normal_angle 
            self.Drone_YAW_NORMAL_VEC = normal_vector

    ############################################## 目标点记录函数 ##############################################

    # 记录起点位置
    def Record_Start_Point_command(self):
        POS = self.update_drone_position_global()
        YAW = self.sensor_data['yaw']

        point_data = np.array([POS[X], POS[Y], POS[Z], YAW])

        self.RACING_POINTS_COMMAND.append(point_data)
    

    ############################################ 视觉导航 ##############################################

    # 常数偏移
    def constant_drift_in_Y(self):

        Drift_Direction_DroneFrame = np.array([0, -1, 0])
        Drift_Direction_WorldFrame = self.Convert_Frame_Drone2World(Drift_Direction_DroneFrame) # 无人机坐标系 -> 世界坐标系

        return Drift_Direction_WorldFrame
    
    # 计算 目标-无人机 距离
    def compute_distance_drone_to_target(self):
        dist = 0.0
        if len(self.target_pos_list_Valid) > 0:
            target_pos = self.target_pos_list_Valid[-1][4]
            drone_pos  = self.Drone_POS_GLOBAL
            dist = np.linalg.norm(target_pos - drone_pos)    # 计算距离
        return dist

    # 系数衰减
    def coefficient_drift_in_Y(self, Dist_Threshold = 1.0, Tensity = 0.8, n = 4):

        x = self.compute_distance_drone_to_target()

        x = np.asarray(x, dtype=float)
        f = np.zeros_like(x)
        mask1 = x >= Dist_Threshold
        mask2 = (x >= 0) & (x < Dist_Threshold)
        f[mask1] = Tensity
        f[mask2] = Tensity * (x[mask2] / Dist_Threshold)**n

        return f # 多项式衰减函数

    # 计算方框偏转角度 -> 偏移速度
    def drift_speed_in_Y(self):

        drift_velocuty = np.array([0, 0, 0]) 

        if self.IMAGE_POINTS_2D is None:
            return np.array([0, 0, 0])

        p0 = self.IMAGE_POINTS_2D[0] # 左上角
        p1 = self.IMAGE_POINTS_2D[1]
        p2 = self.IMAGE_POINTS_2D[2]
        p3 = self.IMAGE_POINTS_2D[3]

        # 左右边
        length_L = np.linalg.norm(p0 - p1) 
        length_R = np.linalg.norm(p2 - p3) 

        # 上下边
        length_T = np.linalg.norm(p0 - p3)
        length_B = np.linalg.norm(p1 - p2)
        length_horizontal = 0.5*(length_T + length_B) # 上下边平均值
        
        # 数据不合法
        if length_horizontal / length_L > 1 or length_horizontal / length_R > 1:
            return np.array([0, 0, 0])

        # 计算角度
        angle_L = np.arccos(length_horizontal / length_L) # 左边角度
        angle_R = np.arccos(length_horizontal / length_R) # 右边角度

        # 
        GAIN = 7

        # 计算偏移速度
        if length_L > length_R:
            drift_velocuty = np.array([0, -1, 0]) * GAIN*(angle_L / np.pi) # 左边偏移速度
            drift_velocuty = self.Convert_Frame_Drone2World(drift_velocuty) # 无人机坐标系 -> 世界坐标系
        else:
            drift_velocuty = np.array([0, +1, 0]) * GAIN*(angle_R / np.pi)
            drift_velocuty = self.Convert_Frame_Drone2World(drift_velocuty) # 无人机坐标系 -> 世界坐标系

        # print(angle_L / np.pi, angle_R / np.pi)

        return drift_velocuty 
        

    # 视觉指令 - 总函数
    def IMG_command(self):

        # 使用该命令需要确保 看到目标 
        direction_target = self.IMAGE_TARGET_VEC_list[4]

        POS = self.update_drone_position_global()
        POS = POS + direction_target * 1.0 + self.drift_speed_in_Y() * self.coefficient_drift_in_Y(Dist_Threshold=1.0, Tensity=0.8) # 目标位置 + 偏移量 #FF0000  
        YAW = self.YAW_TARGET

        command = [POS[X], POS[Y], POS[Z], YAW] # 目标位置 + YAW

        return command

    # 视觉导航
    def get_IMG_command(self):
        # 如果没看到粉色 -> 扫描模式
        if self.IMAGE_POINTS_2D is None:
            control_command = Drone_Controller.Start_Scan_Command() 

        # 如果看到粉色 -> 跟踪模式
        elif (self.IMAGE_POINTS_2D is not None):
            self.scan_FLAG_RESTART = True        # 重启扫描标志位
            control_command = Drone_Controller.IMG_command() # 视觉指令
        
        return control_command

    ############################################### 定位导航 ##############################################
    def stay(self):
        return [self.Drone_POS_GLOBAL[X], self.Drone_POS_GLOBAL[Y], self.Drone_POS_GLOBAL[Z], self.sensor_data['yaw']]

    def get_triangulate_command(self):

        # 直接取 target_pos_list_Valid 中的最后一个点
        if len(self.target_pos_list_Valid) > 0:
            target_pos = self.target_pos_list_Valid[-1][4]
            target_YAW = self.YAW_TARGET
            command = np.append(target_pos, target_YAW) # 目标位置 + YAW
            return command.tolist()
        else:
            return self.stay()


    def get_mix_command(self):
        
        # 默认命令为 视觉命令
        sign = "定位"
        command = self.get_triangulate_command()
        
        # 如果到达目标点
        if self.check_target_AtGate():
            # 到达目标点，使用视觉命令
            command = self.get_IMG_command()
            sign = "视觉"
        
        else:
            # 目标切换，使用三角定位命令
            command = self.get_triangulate_command()

        print("Command: ", sign)
        
        return command
    ############################################### 扫描模式 ##############################################
    def Generate_Scan_Sequence(self):
        T  = 18
        dt = 0.01

        t_sequence = np.arange(0, T + dt, dt) # 生成时间序列

        YAW_shift    = 20 * deg2arc     # 扫描 震荡角度
        delta_YAW    = 360 * deg2arc    # 扫描 震荡角度
        delta_height = 0.0              # 扫描 震荡高度

        omega = 2*np.pi/T

        self.scan_index        = 0    # 初始化索引
        self.scan_FLAG_RESTART = True # 初始化重启标志 
        self.scan_max_index    = T/dt
        self.squence_Shift_YAW    = np.sin( omega * t_sequence) * delta_YAW / 2  + YAW_shift
        self.squence_Shift_Height = np.sin( omega * t_sequence) * delta_height / 2
    
    def Start_Scan_Command(self):

        # 如果上一个状态不是 Scan，则记录 当前 POS + YAW，作为扫描
        if self.scan_FLAG_RESTART:
            self.scan_index = 0
            self.scan_FLAG_RESTART = False

            self.scan_POS = self.update_drone_position_global()
            self.scan_YAW = self.sensor_data['yaw']
        
        # 周期性扫描
        if self.scan_index >= self.scan_max_index:
            self.scan_index = 0

        command = [self.scan_POS[X],
                   self.scan_POS[Y],
                   self.scan_POS[Z] + self.squence_Shift_Height[self.scan_index], # 高度
                   self.scan_YAW    + self.squence_Shift_YAW[self.scan_index]]
        
        self.scan_index += 1

        return command

    ############################################### 基于位置的 巡航模式 ##############################################
    def get_Racing_command_POS_BASED(self):
        
        
        # 如果在 index 内
        try:
            command = self.RACING_PATH[self.RACING_INDEX]

            try: 
                pos1 = self.RACING_PATH[self.RACING_INDEX]
                pos2 = self.RACING_PATH[self.RACING_INDEX + 1]

                yaw = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) # 计算 YAW 角度

                command[3] = yaw # 目标位置 + YAW

            except IndexError:
                yaw = self.sensor_data['yaw'] # 当前 YAW 角度
                command[3] = yaw              # 目标位置 + YAW

            command = command.tolist() # 转换为列表

            # self.RACING_INDEX += 1

        # 如果不在 index 内
        except IndexError:
            command = self.stay()
        
        
        # 控制命令一致性
        Current_Pos = self.Drone_POS_GLOBAL
        Target_Pos  = command[0:3]
        distance = compute_distance(Current_Pos, Target_Pos) # 计算距离

        if distance < 1.5: # 到达目标点范围
            self.RACING_INDEX += 1

        return command


    #  ############################################### 基于时间的 巡航模式 ##############################################
    def get_Racing_command_TIME_BASED(self, dt):

        # 初始化
        if self.timer is None:
            self.timer = 0.0
            self.index_current_setpoint = 0

        # 初始化后
        if self.timer is not None:
            
            # 计算目标位置
            # path_points 索引没用完
            try:
                # Update new setpoint
                if self.timer >= self.racing_time[self.index_current_setpoint]:
                    self.index_current_setpoint += 1
                current_setpoint = self.racing_path[self.index_current_setpoint,:]
            
            # path_points 索引用完了
            except IndexError:
                current_setpoint = self.racing_path[-1]


            # 计算 目标YAW
            try: 
                pos1 = self.RACING_PATH[self.index_current_setpoint]
                pos2 = self.RACING_PATH[self.index_current_setpoint + 1]

                yaw = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) # 计算 YAW 角度

                current_setpoint[3] = yaw # 目标位置 + YAW

            except IndexError:
                yaw = self.sensor_data['yaw'] # 当前 YAW 角度
                current_setpoint[3] = yaw              # 目标位置 + YAW
            
            # 更新路径时间
            self.timer += dt
                    
        return current_setpoint.tolist() # 转换为列表









# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt,           # dt
                ):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    #0000FF 当前控制命令
    global Drone_Controller, Total_Time, Draw, Explore_State

    Total_Time += dt # 累计时间

    # 判断是否第一次运行
    if Drone_Controller is None:
        Drone_Controller = Class_Drone_Controller(sensor_data, camera_data)  # 创建无人机控制器对象
        print("Drone_Controller Created")

    # 无人机状态更新
    Drone_Controller.update(sensor_data, camera_data) 

    # 起飞命令
    if sensor_data['z_global'] < 2.0 and not Drone_Controller.takeoff:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'] + 1.0, sensor_data['yaw']]
        if sensor_data['z_global'] > 1.2:
            Drone_Controller.takeoff = True
        return control_command
        
    # 在探索中 #FF0000
    if Explore_State == 0: # 探索状态
        control_command = Drone_Controller.get_IMG_command()

        # 探索完毕标志位
        if Drone_Controller.AT_GATE and Drone_Controller.RACING_EXPLORE == 5 or Total_Time > 25.0:

            # 修改标志位
            Explore_State = 1

            # 保存数据
            save_data(Drone_Controller.target_pos_list_buffer, file_name="target_positions")

            # 数据处理
            data = AggregatedExtractor(Drone_Controller.target_pos_list_buffer) # 目标点数据处理
            # points = data.convert_to_planning()
            # points = data.convert_to_planning_shift(0.2)                  # 使用偏移数据竞速
            points = data.convert_to_planning_shift_time_customized(0.2)  # 使用偏移数据竞速

            # 根据目标点创建路径点顺序
            # 重构 path，将 Gate 5 移植首位作为起点，并且再添加 Gate 5 作为终点
            path_points = []
            path_points.append(Drone_Controller.Drone_POS_GLOBAL.tolist()) # 当前位置
            # path_points.append(points[-1])    # P5
            path_points.append([1, 4, 1])     # 回到起点
            path_points.extend(points[0:-1])  # P1 -> P4
            path_points.append(points[-1])    # P5
            path_points.append([1, 4, 1])     # 回到起点
            path_points.extend(points)        # P1 -> P5
            path_points.append([1, 4, 1])     # 回到起点
            path_points.extend(points)        # P1 -> P5 # 添加第三圈防止出事
            path_points.append([1, 4, 1])     # 回到起点

            # 基于位置的路径规划
            planner = MotionPlanner3D(obstacles=None, path=path_points)
            Drone_Controller.RACING_PATH = planner.trajectory_setpoints

            # 测试基于的时间路径
            Drone_Controller.racing_path = planner.trajectory_setpoints
            Drone_Controller.racing_time = planner.time_setpoints


     
    # 探索完毕 #FF0000
    elif Explore_State == 1: # 探索完毕
        control_command = Drone_Controller.get_Racing_command_POS_BASED() # 路径规划命令
        # control_command = Drone_Controller.get_Racing_command_TIME_BASED(dt) # 路径规划命令


    return control_command 








############################################ 定义 Filter 类 ############################################
import numpy as np
from collections import defaultdict

class AggregatedExtractor:
    def __init__(self, data_list, gate_point=(4,4,4), dist_thresh=0.5,
                 angle_range=(45,135), cluster_dist=0.8):
        self.gate_center = np.array(gate_point, float)
        self.T, self.min_ang, self.max_ang = dist_thresh, *angle_range
        self.cluster_dist = cluster_dist

        # 输入数据：每个元素为 shape (5,3) 的 np.ndarray，分别对应 P0..P4
        self.data_list = data_list
        # 提取所有 P4 作为中心点序列
        self.points4 = np.array([d[4] for d in data_list], float)

        # 自建数据存储
        self.data_filtered              = None  # 聚合后点 + 方向
        self.data_filtered_sorted       = None  # 排序后的聚合结果
        self.data_filtered_sorted_shift = None  # 平移调整后的坐标

        # 生成扇区划分，并立即执行聚合流程
        self.generate_sector_angles()
        self.compute_conditional_idxs()
        self.sort_aggregated()

    def compute_mask(self):
        # 判断相邻 P4 点位移动是否小于阈值
        dists = np.linalg.norm(np.diff(self.points4, axis=0), axis=1)
        mask = np.concatenate(([True], dists < self.T))
        return mask

    # 判断角度 + 叉乘，计算索引并聚合每个簇的中心点与平均方向
    def compute_conditional_idxs(self):
        mask = self.compute_mask()
        idxs = []
        arrow_dict = {}
        # 筛选满足距离和角度条件的索引，并计算箭头方向
        for i, pt4 in enumerate(self.points4):
            if not mask[i]:
                continue
            v1 = pt4 - self.gate_center
            # P0 和 P3
            p0, p3 = self.data_list[i][0], self.data_list[i][3]
            # 计算摄像机光轴方向
            theta = np.arctan2(*(p0[:2] - p3[:2])[::-1]) - np.pi/2
            v2 = np.array([np.cos(theta), np.sin(theta), 0.])
            # 计算夹角
            cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
            ang = np.degrees(np.arccos(cosang))
            if self.min_ang <= ang <= self.max_ang:
                idxs.append(i)
                # 如果 v1 与 v2 的叉乘在 z 轴为负，则反转方向
                cross_z = np.cross(v1, v2)[2]
                arrow_dict[i] = -v2 if cross_z < 0 else v2

        # 基于空间距离进行聚类
        clusters, curr = [], [idxs[0]] if idxs else []
        for a, b in zip(idxs, idxs[1:]):
            if np.linalg.norm(self.points4[b] - self.points4[a]) < self.cluster_dist:
                curr.append(b)
            else:
                clusters.append(curr)
                curr = [b]
        if curr:
            clusters.append(curr)

        # 聚合每个簇的中心点与平均箭头方向
        agg = []
        for cl in clusters:
            pts = self.points4[cl]
            center = pts.mean(axis=0)
            arrows = np.array([arrow_dict[i] for i in cl])
            avg_arrow = arrows.mean(axis=0)
            agg.append({'Point': center, 'Arrow': avg_arrow})

        self.data_filtered = agg
        return agg

    ############################################## 扇区划分与排序 #######################################
    def generate_sector_angles(self):
        gate, none = np.deg2rad(35), np.deg2rad(25)
        base = -np.pi - gate/2 + gate + none
        self.sectors = [(base + i*(gate + none), base + i*(gate + none) + gate) for i in range(5)]

    def _sector(self, pt2d):
        ang = np.arctan2(*(pt2d - self.gate_center[:2])[::-1])
        for idx, (s, e) in enumerate(self.sectors):
            if s <= ang <= e:
                return f'Gate{idx}', idx
        return None, None

    def sort_aggregated(self):
        groups = defaultdict(list)
        for item in self.data_filtered:
            label, idx = self._sector(item['Point'][:2])
            if idx is not None:
                groups[idx].append((item['Point'], item['Arrow']))

        result = []
        for idx in sorted(groups):
            pts, arrs = zip(*groups[idx])
            mean_pt = np.mean(pts, axis=0)
            mean_arr = np.mean(arrs, axis=0)
            result.append((f'Gate{idx}', mean_pt, mean_arr))

        self.data_filtered_sorted = result

    def convert_to_planning(self):
        # 返回 [(x,y,z), ...]
        return [tuple(np.round(pt, 3)) for _, pt, _ in self.data_filtered_sorted]

    def convert_to_planning_shift(self, shift = 0.3):
        self.data_filtered_sorted_shift = []
        for label, center, arrow in self.data_filtered_sorted:
            angle = np.arctan2(arrow[1], arrow[0])
            new_dir = angle - np.pi/2
            new_vec = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
            new_pt = center + shift * new_vec
            point = tuple(np.round(new_pt, 3))
            self.data_filtered_sorted_shift.append((label, point, arrow))
        return [pt for _, pt, _ in self.data_filtered_sorted_shift]

    def convert_to_planning_shift_time_customized(self, shift = 0.3):
        self.data_filtered_sorted_shift = []
        for label, center, arrow in self.data_filtered_sorted:
            angle = np.arctan2(arrow[1], arrow[0])
            new_dir = angle - np.pi/2
            new_vec = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
            new_pt = center + shift * new_vec
            new_pt[2] -= 0.2 # 基于时间导航需要降低高度
            point = tuple(np.round(new_pt, 3))
            self.data_filtered_sorted_shift.append((label, point, arrow))
        return [pt for _, pt, _ in self.data_filtered_sorted_shift]
    


##################################################### 定义路径规划类 ##############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MotionPlanner3D():
    
    #Question: SIMON PID, what is vel_max set for PID? Check should be same here
    def __init__(self, obstacles, path, DEBUG = False):

        self.path = path

        self.DEBUG = DEBUG

        self.trajectory_setpoints = None

        self.init_params(self.path)

        self.run_planner(obstacles, self.path) # 计算所有数据

        # ---------------------------------------------------------------------------------------------------- ##
    #00FF00 #00FF00
    # 根据起点，终点，以及经过点规划轨迹
    def run_planner(self, obs, path_waypoints):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints
         ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##
    
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs, obs, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def init_params(self, path_waypoints):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        #00FF00
        self.disc_steps = 20    # Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim    = 7.0   # Velocity limit of the drone (m/s)
        self.acc_lim    = 50.0  # Acceleration limit of the drone (m/s²)
        t_f             = 30    # Final time at the end of the path (s)

        # Determine the number of segments of the path
        self.times = np.linspace(0, t_f, len(path_waypoints)) # The time vector at each path waypoint to traverse (Vector of size m) (must be 0 at start)

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        
        # TASK: Fill in the constraint factor matrix values where each row corresponds to the positions, velocities, accelerations, snap and jerk here
        # SOLUTION ---------------------------------------------------------------------------------- ## 
        
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1],            # pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], # vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0],      # acc  
            [60*(t**2), 24*t, 6, 0, 0, 0],             # jerk
            [120*t, 24, 0, 0, 0, 0]                    # snap
        ])

        return A_m

    def compute_poly_coefficients(self, path_waypoints):
        
        
        seg_times = np.diff(self.times) # The time taken to complete each path segment
        m = len(path_waypoints)         # Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs, obs, path_waypoints):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # Uses the class features: self.disc_steps, self.times, self.poly_coeffs, self.vel_lim, self.acc_lim
        x_vals, y_vals, z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        v_x_vals, v_y_vals, v_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        a_x_vals, a_y_vals, a_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))

        # Define the time reference in self.disc_steps number of segements
        time_setpoints = np.linspace(self.times[0], self.times[-1], self.disc_steps*len(self.times))  # Fine time intervals

        # Extract the x,y and z direction polynomial coefficient vectors
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        for i,t in enumerate(time_setpoints):
            seg_idx = min(max(np.searchsorted(self.times, t)-1,0), len(coeff_x) - 1)
            # Determine the x,y and z position reference points at every refernce time
            x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_x[seg_idx*6:(seg_idx+1)*6])
            y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_y[seg_idx*6:(seg_idx+1)*6])
            z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z velocities at every reference time
            v_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_x[seg_idx*6:(seg_idx+1)*6])
            v_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_y[seg_idx*6:(seg_idx+1)*6])
            v_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z accelerations at every reference time
            a_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_x[seg_idx*6:(seg_idx+1)*6])
            a_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_y[seg_idx*6:(seg_idx+1)*6])
            a_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_z[seg_idx*6:(seg_idx+1)*6])

        yaw_vals = np.zeros((self.disc_steps*len(self.times),1))
        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))

        if self.DEBUG:
            self.plot(obs, path_waypoints, trajectory_setpoints)
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        
        # Check that it is less than an upper limit velocity v_lim
        assert vel_max <= self.vel_lim, "The drone velocity exceeds the limit velocity : " + str(vel_max) + " m/s"
        assert acc_max <= self.acc_lim, "The drone acceleration exceeds the limit acceleration : " + str(acc_max) + " m/s²"

        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints
    
    def plot_obstacle(self, ax, x, y, z, dx, dy, dz, color='gray', alpha=0.3):
        """Plot a rectangular cuboid (obstacle) in 3D space."""
        vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])
        
        faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
                [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
                [vertices[j] for j in [0, 3, 7, 4]], [vertices[j] for j in [1, 2, 6, 5]]]
        
        ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha))
    
    def plot(self, obs, path_waypoints, trajectory_setpoints):

        # Plot 3D trajectory
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        if obs is not None:
            for ob in obs:
                self.plot_obstacle(ax, ob[0], ob[1], ob[2], ob[3], ob[4], ob[5])

        ax.plot(trajectory_setpoints[:,0], trajectory_setpoints[:,1], trajectory_setpoints[:,2], label="Minimum-Jerk Trajectory", linewidth=2)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 4)

        # Plot waypoints
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', marker='o', label="Waypoints")

        # Labels and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Motion planning trajectories")
        ax.legend()

        # 俯视图：elev=90（俯视），azim= -90(调整朝向，可根据需要改成0、180等)
        ax.view_init(elev=90, azim=-90)

        plt.show()

