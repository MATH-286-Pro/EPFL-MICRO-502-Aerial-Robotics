import numpy as np
import time
import cv2


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


# 用户定义全局变量
Drone_Controller = None

########################################## 自定基础函数 ##########################################

# 单位化函数
def Unit_Vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # 返回原始向量
    else:
        return v / norm  # 返回单位化向量


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


        # 无人机信息
        self.sensor_data     = None  # 无人机传感器数据
        self.camera_data     = None  # 相机数据
        self.camera_data_BGR = None  # 相机数据 BGR

        # 实时数据 (大写代表实时数据)
        self.Drone_POS_GLOBAL      = None  
        self.Camera_POS_GLOBAL     = None
        self.Drone_TARGET_VEC      = None  # 目标方向
        self.Drone_TARGET_VEC_list = []  # 目标方向列表，0-3为矩形的四个点，4为中心点

        self.Drone_YAW_TARGET    = None
        self.Drone_YAW_NORMAL    = None  

        # 缓存数据 (三角定位)
        self.Drone_Pos_Buffer             = [] # 位置缓存
        self.Drone_Target_Vec_Buffer      = [] # 方向缓存
        self.Drone_Target_Vec_List_Buffer = [] # 方向列表缓存
        self.min_cumulative_baseline = 0.3  # 设定累计基线距离阈值

        self.valid_pos_buffer  = [] # 有效位置缓存
        self.valid_vec_buffer  = [] # 有效方向缓存
        self.target_pos_buffer = [] # 有效目标缓存

        self.target_pos_list_buffer = [] # 有效目标缓存列表

        # 路径数据记录
        self.RACING_START_POINT  = None
        self.RACING_MID_POINT    = None

        self.EXAMING             = True
        self.RACING_EXPLORE      = 0     
        self.RACING_EXPLORE_DONE = False # 是否完成探索，默认开始能检测到第一个点

        self.RACING_BEGIN_INDEX        = [0] # 记录索引，用于记录 某个Gate 起始 index
        self.RACING_END_INDEX          = []  # 记录索引，用于记录 某个Gate 结束 index

        # 视觉命令锁
        self.LOCK = False
        self.IMG_LAST_DIRECTION = None # 上一个视觉方向

        # 启动标志
        self.START_FLAG   = False  

        # 启动记录
        self.update(sensor_data, camera_data) # 更新数据
        self.IMG_direction_Initialize()       # 初始化方向

        # 记录起始位置
        self.Record_Start_Point()

        # 生成扫描偏移序列
        self.Generate_Scan_Sequence() 

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
        self.update_drone_quat()               # 更新无人机四元数
        self.update_drone_position_global()    # 更新无人机坐标
        self.update_camera_position_global()   # 更新相机坐标
        self.update_img2vector_list(DEBUG = True)     # 相机坐标系下目标位置列表

        # 检查 + 记录目标点
        self.Record_Mid_Point()                # 记录中间点

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
    
    def check_target_arrived(self):
        
        try:
            target_pos = self.target_pos_list_buffer[-1][4]  # 目标位置
            drone_pos  = self.update_drone_position_global() # 无人机位置

            dist = np.linalg.norm(target_pos - drone_pos)    # 计算距离

            print("dist", dist, "Target", target_pos)

            if dist <= 0.5 and self.EXAMING: 
                self.EXAMING = False
                return True
            else:
                return False
        except IndexError:
            return False
    
    def check_target_switch(self):

        try:
            target_pos_1 = self.target_pos_list_buffer[-2][4]  # 目标位置
            target_pos_2 = self.target_pos_list_buffer[-1][4]  # 目标位置

            delta = np.linalg.norm(target_pos_2 - target_pos_1) # 计算增量

            if delta >= 2.0: # 检测到下一个点
                self.EXAMING = True
                return True
            else:
                return False
            
        except IndexError:
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

    ########################################## 图像处理函数 ##########################################

    # 图像 -> 粉色 mask
    def img_BGR_to_PINK(self, DEBUG = False):

        bgr = self.camera_data_BGR.copy()

        # OpenCV 是 BGR 顺序！
        upper_pink  = np.array([255, 185, 255])  # B, G, R
        lower_pink  = np.array([190, 60, 190])   
        binary_mask = cv2.inRange(bgr, lower_pink, upper_pink)

        # 可视化 mask 和提取结果
        if DEBUG:
            cv2.imshow("Gray", binary_mask)                              # 灰度图 / mask
            pink_only = cv2.bitwise_and(bgr, bgr, mask=binary_mask)      # 应用 mask 扣图
            cv2.imshow("Pink", pink_only)                                # 粉色图

        return binary_mask

    # 图像 -> 特征点
    def img_to_points(self, binary_mask, DEBUG = False):
        
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
            largest_rect = np.squeeze(largest_rect, axis=1)                     # 将 4x1x2 的数组转换为 4x2 的数组
            rect_center  = compute_target_center(largest_rect)                  # np.Float64
            target_rect  = np.append(largest_rect, [rect_center], axis = 0)     # 添加中心点

            # 在原图上画出四个点
            if DEBUG:
                length = len(target_rect)
                increment = int(255/(length+1))  # 计算增量
                green_value = increment
                for x, y in target_rect:
                    cv2.circle(Feature_Frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.imshow("Rectangle Corners", Feature_Frame)

            self.IMAGE_POINTS = target_rect.copy() # 复制点 #00FF00 #00FF00

            return target_rect # 返回拟合的 四边形顶点 + 中心点
        
        else:
            if DEBUG:
                cv2.imshow("Rectangle Corners", Feature_Frame)

            self.IMAGE_POINTS = None # 没有找到目标
            return None


    # 图像 -> 方向向量列表
    def update_img2vector_list(self, DEBUG = False):

        # 初始化
        Vector_Cam2Target_WorldFrame_list = []

        # 图像处理
        cv2.waitKey(1) # 如果放在 return 后面会报错
        binary_mask = self.img_BGR_to_PINK()                 # 抠图
        cam_points  = self.img_to_points(binary_mask, DEBUG) # 计算特征点
        
        # 计算向量
        if cam_points is not None:
            
            for cam_point in cam_points:
                # 目标方向：相机坐标系 -> 无人机坐标系
                Vector_Cam2Target_DroneFrame = self.Convert_Frame_Cam2Drone(cam_point)    
                Vector_Cam2Target_DroneFrame = Unit_Vector(Vector_Cam2Target_DroneFrame)  

                # 目标方向：无人机坐标系 -> 世界坐标系
                Vector_Cam2Target_WorldFrame = self.Convert_Frame_Drone2World(Vector_Cam2Target_DroneFrame)
                Vector_Cam2Target_WorldFrame = Unit_Vector(Vector_Cam2Target_WorldFrame)  

                Vector_Cam2Target_WorldFrame_list.append(Vector_Cam2Target_WorldFrame)    # 添加到列表中

            self.Drone_TARGET_VEC_list = Vector_Cam2Target_WorldFrame_list 

        else :
            self.Drone_TARGET_VEC_list = None


    ########################################## 三角定位部分 ######################################################
    def target_positioning(self,
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
    def Compute_Target_List_with_Buffer(self):

        # 如果视野内无目标，不能将 None 添加到缓存中
        # #0000FF 大写代表实时更新数据，实时数据会包含 None #0000FF
        if (self.Drone_POS_GLOBAL is not None) and (self.Drone_TARGET_VEC_list is not None):

            # 更新 Buffer
            self.Drone_Pos_Buffer.append(self.Drone_POS_GLOBAL)
            self.Drone_Target_Vec_List_Buffer.append(self.Drone_TARGET_VEC_list)
        
        # 计算累计位移
        if len(self.Drone_Pos_Buffer) >= 2: # 至少两帧数据
            cumulative_baseline = np.linalg.norm(self.Drone_Pos_Buffer[-1] - self.Drone_Pos_Buffer[0])
            
            if cumulative_baseline >= self.min_cumulative_baseline:
                
                # 初始化
                Target_Pos_list = np.zeros((5,3)) # 5 个目标位置

                # 计算 5 个目标位置
                for i in range(5):
                    Target_Pos_list[i] = self.target_positioning(self.Drone_Pos_Buffer[-1], 
                                                                 self.Drone_Pos_Buffer[0],
                                                                 self.Drone_Target_Vec_List_Buffer[-1][i],
                                                                 self.Drone_Target_Vec_List_Buffer[0][i])

                # print("目标全局坐标：",Target_Pos_list[-1], self.Drone_Target_Vec_List_Buffer[-1][4])

                # 缓存清空 #FF0000 需要修改逻辑，充分利用数据
                self.Drone_Pos_Buffer             = [] # 清空缓存
                self.Drone_Target_Vec_List_Buffer = []

                # # 缓存保留部分数据
                # length = len(self.Drone_Pos_Buffer)
                # self.Drone_Pos_Buffer = [self.Drone_Pos_Buffer[length - 2]]                         # 保留帧数据
                # self.Drone_Target_Vec_List_Buffer = [self.Drone_Target_Vec_List_Buffer[length - 2]] # 保留帧数据

                # 更新目标值
                self.target_pos_list_buffer.append(Target_Pos_list) # 目标缓存

                #FF0000 测试不同 YAW 值
                self.Compute_YAW_TARGET() # 计算目标 YAW
                self.Compute_YAW_NORMAL() 



    ############################################# 计算目标 YAW #############################################
    def Compute_YAW_TARGET(self):
        Vector_3D = self.Drone_TARGET_VEC_list[4]
        self.Drone_YAW_TARGET = np.arctan2(Vector_3D[1], Vector_3D[0])  # 计算 YAW 角度
    
    def Compute_YAW_NORMAL(self):

        # points_global = self.target_pos_list_buffer[-1]
        points_global = np.mean(self.target_pos_list_buffer, axis=0) # 计算平均值

        vec1 = points_global[0] - points_global[3]
        vec2 = points_global[1] - points_global[2]
        vec = 0.5*(vec1 + vec2) 
        theta = np.arctan2(vec[1], vec[0])  # 计算 YAW 角度

        normal_angle = theta - np.pi / 2
        normal_vector = np.array([np.cos(normal_angle), np.sin(normal_angle), 0]) #FF0000 注意这个是不精确的法向量

        self.Drone_YAW_NORMAL = normal_angle 
        self.Drone_YAW_NORMAL_VEC = normal_vector

    ############################################## 目标点记录函数 ##############################################

    # 记录起点位置
    def Record_Start_Point(self):
        POS = self.update_drone_position_global()
        YAW = self.sensor_data['yaw']
        self.RACING_START_POINT = [POS[X], POS[Y], 1.0, YAW] 
    
    # 记录 Gate 位置
    def Record_Mid_Point(self):
        
        if self.check_target_switch():
            self.RACING_BEGIN_INDEX.append(len(self.target_pos_list_buffer) - 1) # 记录索引
        
        if self.check_target_arrived():
            self.RACING_END_INDEX.append(len(self.target_pos_list_buffer))
            self.RACING_EXPLORE += 1
            print("Gate %d Arrived" % self.RACING_EXPLORE)
        
        if self.RACING_EXPLORE >= 5:
            self.RACING_EXPLORE_DONE = True
        
    ############################################## 轨迹生成函数 ##############################################

    def Generate_Trajectory(self, T = 3, dt = 0.1):
        p0 = self.Camera_POS_GLOBAL
        pT  = self.target_pos_list_buffer[0][4] + self.Drone_YAW_NORMAL_VEC * 0.8 # 使用第一个目标点测试

        v0 = np.array([self.sensor_data['v_x'],
                        self.sensor_data['v_y'],
                        self.sensor_data['v_z']])
        vT  = self.Drone_YAW_NORMAL_VEC

        # 根据起始条件构建三次多项式的系数
        a0 = p0
        a1 = v0
        Delta = pT - p0 - v0 * T
        a2 = (3 * Delta) / (T ** 2) - (2 * v0 + vT) / T
        a3 = (-2 * Delta) / (T ** 3) + (v0 + vT) / (T ** 2)

        # 生成轨迹数据点
        t_vals = np.arange(0, T + dt, dt)
        trajectory = np.array([a0 + a1 * t + a2 * t**2 + a3 * t**3 for t in t_vals])

        self.test_trajectory     = trajectory.copy() # 测试轨迹
        self.test_trajectory_idx = 0

        return trajectory


    ############################################ 基于图像控制器 ##############################################

    # 防止开始 Lock 导致的方向错误
    def IMG_direction_Initialize(self):
        self.IMG_LAST_DIRECTION = self.Drone_TARGET_VEC_list[4] # 目标方向

    def IMG_direction(self):

        direction = self.Drone_TARGET_VEC_list[4] # 目标方向

        if self.LOCK:
            return self.IMG_LAST_DIRECTION
        
        if not self.LOCK:
            self.IMG_LAST_DIRECTION = direction.copy() # 记录上一个方向
            return direction

    def IMG_command(self):

        # 使用该命令需要确保 看到目标 

        delta_POS = self.IMG_direction()
        POS = self.update_drone_position_global()
        POS = POS + delta_POS * 1.0
        YAW = self.Drone_YAW_TARGET

        command = [POS[X], POS[Y], POS[Z], YAW] # 目标位置 + YAW

        return command


    ############################################### 扫描函数 ##############################################
    def Generate_Scan_Sequence(self):
        T  = 4
        dt = 0.01

        t_sequence = np.arange(0, T + dt, dt) # 生成时间序列

        YAW_shift  = 20 * deg2arc    # 扫描 震荡角度
        delta_YAW  = 120 * deg2arc    # 扫描 震荡角度
        delta_height = 0.3           # 扫描 震荡高度

        omega = 2*np.pi/T

        self.scan_index        = 0    # 初始化索引
        self.scan_FLAG_RESTART = True # 初始化重启标志 
        self.scan_max_index    = T/dt
        self.squence_Shift_YAW    = np.sin( omega * t_sequence) * delta_YAW / 2  + YAW_shift
        self.squence_Shift_Height = np.sin( omega * t_sequence) * delta_height / 2
    
    def Start_Scan_Command(self):
        if self.scan_FLAG_RESTART:
            self.scan_index = 0
            self.scan_FLAG_RESTART = False

            self.scan_POS = self.update_drone_position_global()
            self.scan_YAW = self.sensor_data['yaw']
            print("Scan Restart")
        
        if self.scan_index >= self.scan_max_index:
            self.scan_index = 0

        command = [self.scan_POS[X],
                   self.scan_POS[Y],
                   self.scan_POS[Z] + self.squence_Shift_Height[self.scan_index], # 高度
                   self.scan_YAW    + self.squence_Shift_YAW[self.scan_index]]
        
        self.scan_index += 1

        return command













# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt,           # dt
                ):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    #0000FF 当前控制命令
    global Drone_Controller

    # 判断是否第一次运行
    if Drone_Controller is None:
        Drone_Controller = Class_Drone_Controller(sensor_data, camera_data)  # 创建无人机控制器对象
        print("Drone_Controller Created")

    # 无人机状态更新
    Drone_Controller.update(sensor_data, camera_data) 
    Drone_Controller.Compute_Target_List_with_Buffer() # 测试 list 更新

    # Take off example
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    # ---- YOUR CODE HERE ----

    #FF0000         
    # 如果没看到粉色 
    if Drone_Controller.IMAGE_POINTS is None:
        control_command = Drone_Controller.Start_Scan_Command() 
        print(1)

    # 如果看到粉色 + 离目标点较远 + 未切换目标
    elif (Drone_Controller.IMAGE_POINTS is not None):

        Drone_Controller.scan_FLAG_RESTART = True        # 重启扫描标志位

        # 距离 Gate 较远 
        # 需要解锁
        # 视觉导航
        if (not Drone_Controller.check_target_arrived()):
            Drone_Controller.LOCK = False
            control_command = Drone_Controller.IMG_command()
            print(2)

        # 距离 Gate 较近 
        # 需要上锁
        # 上锁导航
        elif (Drone_Controller.check_target_arrived()):
            Drone_Controller.LOCK = True
            control_command = Drone_Controller.IMG_command()
            print(3)


        # 未知情况
        else:
            control_command = [sensor_data['x_global'], 
                              sensor_data['y_global'], 
                              sensor_data['z_global'],
                              sensor_data['yaw']]
            print(5)

    print(Drone_Controller.check_target_arrived(), Drone_Controller.check_target_switch(), Drone_Controller.RACING_EXPLORE)

    return control_command 