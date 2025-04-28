import numpy as np
import time
import cv2

from assignment.filter import *
from assignment.planning import *
from assignment.base import *


# 宏定义
X = 0 # 四元数下标
Y = 1
Z = 2
W = 3  

# 用户定义全局变量
Drone_Controller = None
Total_Time       = 0
Draw             = False # 是否绘制过轨迹
Explore_State    = 0     # 0 代表在探索中，1 代表探索完毕


# 定义无人机类
class Class_Drone_Controller:

    def __init__(self, sensor_data, camera_data):

        # 相机参数
        self.f_pixel = 161.013922282   # 相机焦距
        self.vector_Drone2Cam_DroneFrame = np.array([0.03,0.00,0.01]) # 无人机中心到相机偏移向量
        self.camera_size = [300,300]
        self.cam_center_x = self.camera_size[X] / 2 # 像素中心点 x
        self.cam_center_y = self.camera_size[Y] / 2 # 像素中心点 y

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
        self.target_aro_list_Valid  = [] # 目标法向量         [配合 Valid Pos]

        # 路径数据记录
        self.AT_GATE             = False # 是否到达 Gate
        self.RACING_EXPLORE      = 0     
        self.RACING_POINT_INDEX  = [] # 记录索引，用于记录 某个Gate 起始 index

        # 
        self.lap_start  = False
        self.lap_finish = False
        self.lap_index  = 0
        self.lap_path   = None
        self.lap_time   = None
        self.timer      = 0

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
        #    self.update_IMAGE_TO_POINTS_2D  # 更新相机 2D 坐标位置

        # 更新 三角定位 4+1 列表
        self.update_Target_List_with_Buffer()               # 更新目标点列表 [slef.target_pos_list_buffer] 列表数据
        #  self.update_Target_list_Filtered_CallBack()      # 数据滤波
        #    self.check_target_switch() # 检测目标切换        # 是否切换目标

        # 检测
        self.check_target_AtGate()                       # 检测目标点是否到达
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

                    # print("到达目标点范围！")

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
                # print(self.RACING_EXPLORE-1, "->", self.RACING_EXPLORE, "目标点切换！")  

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
                    # print(self.RACING_EXPLORE-1, "->", self.RACING_EXPLORE, "目标点切换！") 

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
                data = self.target_pos_list_buffer[-1]    # 目标点 4+1 缓存 #00FF00
                p0 = data[0]
                p3 = data[3]
                theta  = np.arctan2(p0[Y] - p3[Y], p0[X] - p3[X]) - np.pi / 2
                vector = np.array([np.cos(theta), np.sin(theta), 0])

                self.target_pos_list_Valid.append(data)    # 更新目标位置
                self.target_aro_list_Valid.append(vector)  # 更新目标法向量


                self.check_target_switch() # 检测目标切换

    ############################################### 计算目标 YAW #############################################
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


    ############################################### 视觉导航 ##############################################

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
        POS = POS + direction_target * 1.0 + self.drift_speed_in_Y() * self.coefficient_drift_in_Y(Dist_Threshold=1.0, Tensity=0.7) # 目标位置 + 偏移量 #FF0000  
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
        if not self.check_target_AtGate():
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
        # T  = 18
        T  = 6 
        dt = 0.01

        t_sequence = np.arange(0, T + dt, dt) # 生成时间序列

        YAW_shift    = np.deg2rad(20)    # 扫描 震荡角度
        delta_YAW    = np.deg2rad(360)   # 扫描 震荡角度
        delta_height = 0.0               # 扫描 震荡高度

        omega = 2*np.pi/T

        self.scan_index        = 0    # 初始化索引
        self.scan_FLAG_RESTART = True # 初始化重启标志 
        self.scan_max_index    = T/dt

        # self.sequence_Shift_YAW    = np.sin( omega * t_sequence) * delta_YAW / 2  + YAW_shift  # 正弦函数扫描
        self.sequence_Shift_YAW    =  (t_sequence) / T * (2 * np.pi) + YAW_shift                 # 匀速扫描

        self.sequence_Shift_Height = np.sin( omega * t_sequence) * delta_height / 2              # 当前高度上下扫描
        # self.sequence_Shift_Height = np.ones_like(t_sequence)                                    # 固定高度扫描
    
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
                   np.maximum(self.scan_POS[Z] + self.sequence_Shift_Height[self.scan_index], 0.4), # 高度 最低为 0.4
                   self.scan_YAW    + self.sequence_Shift_YAW[self.scan_index]]
        
        self.scan_index += 1

        return command

    ############################################### 巡航模式 ##############################################
    def path_command_init(self, path, time = None):
        self.lap_start  = True
        self.lap_finish = False
        self.lap_index  = 0
        self.lap_path   = path
        self.lap_time   = time
        self.timer      = 0
    
    def return_path_command(self, mode = "position", dt = None, YAW_SHIFT = 0):

        speed = 1.0

        if self.lap_start == False:
            self.lap_start = True
            self.lap_index = 0
            self.lap_finish = False

        # 基于位置
        if mode == "position":
            try:
                command = self.lap_path[self.lap_index]

                try:
                    pos1 = self.lap_path[self.lap_index]
                    pos2 = self.lap_path[self.lap_index+1]
                    yaw  = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
                    command[3] = yaw + YAW_SHIFT
                except IndexError:
                    yaw = self.sensor_data["yaw"]
                    command[3] = yaw
            except IndexError:
                command = self.stay()
            
            current_pos = self.Drone_POS_GLOBAL
            target_pos  = command[0:3]
            distance = compute_distance(current_pos, target_pos)

            if distance < speed:
                self.lap_index += 1
                if self.lap_index == len(self.lap_path):
                    self.lap_start  = False
                    self.lap_finish = True
                    self.lap_index  = 0
            
            return command.tolist()
        
        # 基于时间
        elif mode == "time":
            # self.lap+path 索引没用完
            try:
                if self.timer >= self.lap_time[self.lap_index]:
                    self.lap_index += 1
                command = self.lap_path[self.lap_index]
            
            except IndexError:
                command = self.lap_path[-1]
                self.lap_start  = False
                self.lap_finish = True
            
            try:
                pos1 = self.lap_path[self.lap_index]
                pos2 = self.lap_path[self.lap_index+1]
                yaw  = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
                command[3] = yaw + YAW_SHIFT
            except IndexError:
                yaw = self.sensor_data["yaw"]
                command[3] = yaw
            
            self.timer += dt

            return command.tolist()
        
        else:
            raise ValueError("Invalid mode. Use 'position' or 'time'.")


    def get_path_command(self, path, time, mode = "position", dt = None, YAW_SHIFT = 0):
        
        # 初始化
        if self.lap_start == False:
            self.path_command_init(path = path,
                                   time = time)
        
        # 生成命令
        if self.lap_start == True and mode == "position":
            return self.return_path_command(mode = "position", dt = None, YAW_SHIFT = YAW_SHIFT) 
        elif self.lap_start == True and mode == "time":
            return self.return_path_command(mode = "time", dt = dt, YAW_SHIFT = YAW_SHIFT)
        else:
            return self.stay()



    ############################################### 点位排序 ############################################
    def return_path_order_normal(self, gate_points):
        path_points = []
        path_points.append(self.Drone_POS_GLOBAL.tolist()) # 当前位置
        path_points.append([1, 4, 1])     # 回到起点
        path_points.extend(gate_points[0:-1])  # P1 -> P4
        path_points.append(gate_points[-1])    # P5
        path_points.append([1, 4, 1])     # 回到起点
        path_points.extend(gate_points)        # P1 -> P5
        path_points.append([1, 4, 1])     # 回到起点
        path_points.extend(gate_points)        # P1 -> P5 # 添加第三圈防止出事
        path_points.append([1, 4, 1])     # 回到起点

        return path_points

    def return_path_order_xunhang(self,gate_points):

        path_points = []
        path_points.append(self.Drone_POS_GLOBAL.tolist()) # 当前位置

        path_points.extend(gate_points)
        path_points.append([1, 4, 1])     # 回到起点

        path_points.extend(gate_points)
        path_points.append([1, 4, 1])     # 回到起点

        path_points.extend(gate_points)
        path_points.append([1, 4, 1])     # 回到起点

        return path_points







# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt,           # dt
                ):

    global Drone_Controller, Total_Time, Draw, Explore_State

    Total_Time += dt # 累计时间

    # 判断是否第一次运行
    if Drone_Controller is None:
        Drone_Controller = Class_Drone_Controller(sensor_data, camera_data)  # 创建无人机控制器对象
        print("Drone_Controller Created")

        # 路径点
        start = [1, 4, 1] # 起点
        mid   = [[2, 6, 1],
                 [6, 6, 1],
                 [6, 2, 1],
                 [2, 2, 1]]
        path = []
        path.append(start)
        path.extend(mid)
        path.append(start)
        
        # path = [[1, 4, 1],
        #         [1, 1, 1],
        #         [7, 1, 1],
        #         [7, 7, 1],
        #         [1, 7, 1],
        #         [2, 6, 1],   # 绕过中心点
        #         [6, 4, 1],   # 绕过中心点
        #         [0.2, 3, 1]] # 绕过中心点 #00FF00
        planner = MotionPlanner3D(obstacles=None,
                                  time = None, 
                                  path = path)
        Drone_Controller.racing_path = planner.trajectory_setpoints
        Drone_Controller.racing_time = planner.time_setpoints


    # 无人机状态更新
    Drone_Controller.update(sensor_data, camera_data) 

    # 起飞命令
    if sensor_data['z_global'] < 2.0 and not Drone_Controller.takeoff:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'] + 1.0, sensor_data['yaw']]
        if sensor_data['z_global'] > 1.2:
            Drone_Controller.takeoff = True
        return control_command
        
    # 在探索中 #FF0000
    if Explore_State == 0:   # 探索：第一圈
      
        control_command = Drone_Controller.get_path_command(path = Drone_Controller.racing_path,
                                                            time = Drone_Controller.racing_time,
                                                            mode = "position",
                                                            dt   = dt,
                                                            YAW_SHIFT = np.deg2rad(-25))
        if Drone_Controller.lap_finish == True:
            Explore_State += 1 # 修改状态位

    elif Explore_State == 1: # 探索：第二圈

        control_command = Drone_Controller.get_path_command(path = Drone_Controller.racing_path,
                                                            time = Drone_Controller.racing_time,
                                                            mode = "position",
                                                            dt   = dt,
                                                            YAW_SHIFT = np.deg2rad(-40))

        # 探索完毕标志位
        if Drone_Controller.lap_finish == True:

            # 修改标志位
            Explore_State += 1

            # 保存数据
            save_data(Drone_Controller.target_pos_list_buffer, file_name="target_positions")

            # 数据处理
            data        = AggregatedExtractor(Drone_Controller.target_pos_list_buffer) # 目标点数据处理
            gate_points = data.convert_to_planning()
            # gate_points = data.convert_to_planning_shift(0.2)                  # 使用偏移数据竞速
            # gate_points = data.convert_to_planning_shift_time_customized(0.2)  # 使用偏移数据竞速
            # gate_points = data.shift_points_bidirectional(1.0)

            # 路径规划
            planner = MotionPlanner3D(obstacles=None, 
                                      time = None, 
                                      path = Drone_Controller.return_path_order_xunhang(gate_points))
            Drone_Controller.racing_path = planner.trajectory_setpoints
            Drone_Controller.racing_time = planner.time_setpoints

     
    # 探索完毕 #FF0000
    elif Explore_State == 2: # 探索完毕
        control_command = Drone_Controller.get_path_command(path = Drone_Controller.racing_path,
                                                            time = Drone_Controller.racing_time,
                                                            mode = "position",
                                                            dt   = dt)


    return control_command 

