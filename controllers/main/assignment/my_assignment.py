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

# 计算目标法向量
def compute_normal_vector(v1,v2):
    # 说明：从 v1 到 v2 右手定律，大拇指为计算法向量
    normal = np.cross(v1, v2)     # 计算法向量
    normal = Unit_Vector(normal)  # 单位化
    return normal

########################################## 自定基础函数 ##########################################




# 定义无人机类
class Class_Drone_Controller:

    def __init__(self, camera_data):

        # 基本参数
        self.f_pixel = 161.013922282   # 相机焦距
        self.vector_Drone2Cam_DroneFrame = np.array([0.03,0.00,0.01]) # 无人机中心到相机偏移向量

        cam_center_y, cam_center_x, _ = camera_data.shape
        self.cam_center_x = cam_center_x / 2 # 像素中心点 x
        self.cam_center_y = cam_center_y / 2 # 像素中心点 y
        
        # 全局变量 (三角定位)
        self.pos_buffer      = [] # 位置缓存
        self.vec_buffer      = [] # 方向缓存
        self.vec_list_buffer = [] # 方向列表缓存
        self.min_cumulative_baseline = 0.3  # 设定累计基线距离阈值

        self.valid_pos_buffer  = [] # 有效位置缓存
        self.valid_vec_buffer  = [] # 有效方向缓存
        self.target_buffer     = [] # 有效目标缓存

        self.target_pos_list_buffer = [] # 有效目标缓存列表

        # 无人机信息
        self.sensor_data     = None  # 无人机传感器数据
        self.camera_data     = None  # 相机数据
        self.camera_data_BGR = None  # 相机数据 BGR

        # 计算数据
        self.Drone_Pos_Global    = None  
        self.Drone_target_vec    = None  # 目标方向
        self.Drone_target_vec_list = []  # 目标方向列表，0-3为矩形的四个点，4为中心点
        self.Drone_target_normal = None  # 目标矩形法向量
        self.Drone_target_YAW    = None

        # 启动标志
        self.START_FLAG       = False  


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
        self.Drone_Pos_Global = self.get_position_global()         # 无人机全局坐标系下位置
        self.Drone_target_vec = self.img_to_vector(DEBUG=True)     # 相机坐标系下目标位置
        self.Drone_target_vec_list = self.img_to_vector_list()     # 相机坐标系下目标位置列表

    ########################################## 传感器函数 ##########################################
    def get_quat_from_sensor(self):
        quat = np.array([self.sensor_data['q_x'], 
                         self.sensor_data['q_y'], 
                         self.sensor_data['q_z'], 
                         self.sensor_data['q_w']])
        return quat

    def get_position_global(self):
        position = np.array([self.sensor_data['x_global'], 
                             self.sensor_data['y_global'], 
                             self.sensor_data['z_global']])
        return position

    def get_position_cam_global(self):
        P_Drone_global = self.get_position_global()      # 无人机全局坐标系下位置
        Q_Drone2World  = self.get_quat_from_sensor()     # 无人机四元数
        P_Drone2Cam_Shift_global = vector_rotate(self.vector_Drone2Cam_DroneFrame, Q_Drone2World)  # 无人机坐标系下相机位置
        P_Cam_global   = P_Drone_global + P_Drone2Cam_Shift_global       # 相机全局坐标系下位置
        return P_Cam_global
    
    ########################################## 坐标变换函数 ##########################################
    def Convert_Frame_Drone2World(self, P_DroneFrame):

        Q_Drone2World = self.get_quat_from_sensor()  # 无人机四元数
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
                for x, y in target_rect:
                    cv2.circle(Feature_Frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.imshow("Rectangle Corners", Feature_Frame)
            
            return target_rect # 返回拟合的 四边形顶点 + 中心点
        
        else:
            if DEBUG:
                cv2.imshow("Rectangle Corners", Feature_Frame)
            return None


    # 图像 -> 方向向量    
    def img_to_vector(self, DEBUG = False):

        # 定义索引
        index_center = 4

        # 图像处理
        cv2.waitKey(1) # 如果放在 return 后面会报错
        binary_mask = self.img_BGR_to_PINK()                 # 抠图
        target_rect = self.img_to_points(binary_mask, DEBUG) # 计算特征点
        
        # 计算向量
        if target_rect is not None:
            
            # 目标方向：相机坐标系 -> 无人机坐标系
            Vector_Cam2Target_DroneFrame = self.Convert_Frame_Cam2Drone(target_rect[index_center]) # 计算相机坐标系下的目标方向
            Vector_Cam2Target_DroneFrame = Unit_Vector(Vector_Cam2Target_DroneFrame)  # 单位化

            # 目标方向：无人机坐标系 -> 世界坐标系
            Vector_Cam2Target_WorldFrame = self.Convert_Frame_Drone2World(Vector_Cam2Target_DroneFrame)
            Vector_Cam2Target_WorldFrame = Unit_Vector(Vector_Cam2Target_WorldFrame)  # 单位化

            return Vector_Cam2Target_WorldFrame 

        else:
            return None

    # 图像 -> 方向向量列表
    def img_to_vector_list(self, DEBUG = False):

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
                Vector_Cam2Target_DroneFrame = self.Convert_Frame_Cam2Drone(cam_point) # 计算相机坐标系下的目标方向
                Vector_Cam2Target_DroneFrame = Unit_Vector(Vector_Cam2Target_DroneFrame)  # 单位化

                # 目标方向：无人机坐标系 -> 世界坐标系
                Vector_Cam2Target_WorldFrame = self.Convert_Frame_Drone2World(Vector_Cam2Target_DroneFrame)
                Vector_Cam2Target_WorldFrame = Unit_Vector(Vector_Cam2Target_WorldFrame)  # 单位化

                Vector_Cam2Target_WorldFrame_list.append(Vector_Cam2Target_WorldFrame)    # 添加到列表中

            return Vector_Cam2Target_WorldFrame_list 

        else:
            return None


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

        # print("目标全局坐标：",T)

        return T

        # 点1 = [2.12, 1.84, 1.24]
        # 点2 = [5.12, 2.30, 0.78]


    ############################################# 三角定位，缓存更新 #############################################
    def Compute_Target_With_Buffer(self):

        # 如果视野内无目标，不能将 None 添加到缓存中
        if (self.Drone_Pos_Global is not None) and (self.Drone_target_vec is not None):
            self.pos_buffer.append(self.Drone_Pos_Global)
            self.vec_buffer.append(self.Drone_target_vec)

        # 计算累计位移
        if len(self.pos_buffer) >= 2: # 至少两帧数据
            cumulative_baseline = np.linalg.norm(self.pos_buffer[-1] - self.pos_buffer[0])

            if cumulative_baseline >= self.min_cumulative_baseline:
                T = self.target_positioning(self.pos_buffer[-1], 
                                            self.pos_buffer[0],
                                            self.vec_buffer[-1],
                                            self.vec_buffer[0])
                
                # 清空缓存 #FF0000 需要修改逻辑，充分利用数据
                self.valid_pos_buffer.append(self.pos_buffer[-1])  # 有效位置缓存
                self.valid_vec_buffer.append(self.vec_buffer[-1])    # 有效方向缓存

                self.pos_buffer = [] # 清空缓存
                self.vec_buffer   = []

                # self.position_buffer = [self.position_buffer[-1]] # 只保留最新位置
                # self.vector_buffer   = [self.vector_buffer[-1]]   # 只保留最新方向

                # 更新目标值
                self.target_buffer.append(T) # 目标缓存
                self.Compute_YAW() # 计算目标 YAW
                        
        return None
    
    def Compute_Target_List_with_Buffer(self):

        # 如果视野内无目标，不能将 None 添加到缓存中
        if (self.Drone_Pos_Global is not None) and (self.Drone_target_vec_list is not None):
            self.pos_buffer.append(self.Drone_Pos_Global)
            self.vec_list_buffer.append(self.Drone_target_vec_list)
            self.vec_buffer.append(self.Drone_target_vec_list[-1])
        
        # 计算累计位移
        if len(self.pos_buffer) >= 2: # 至少两帧数据
            cumulative_baseline = np.linalg.norm(self.pos_buffer[-1] - self.pos_buffer[0])
            
            if cumulative_baseline >= self.min_cumulative_baseline:
                
                # 初始化
                Target_Pos_list = np.zeros((5,3)) # 5 个目标位置

                # 计算 5 个目标位置
                for i in range(5):
                    Target_Pos = self.target_positioning(self.pos_buffer[-1], 
                                                         self.pos_buffer[0],
                                                         self.vec_list_buffer[-1][i],
                                                         self.vec_list_buffer[0][i])
                    Target_Pos_list[i] = Target_Pos

                # 清空缓存 #FF0000 需要修改逻辑，充分利用数据
                # self.valid_pos_buffer.append(self.pos_buffer[-1])    # 有效位置缓存
                # self.valid_vec_buffer.append(self.vec_buffer[-1])    # 有效方向缓存

                self.pos_buffer      = [] # 清空缓存
                self.vec_list_buffer = []


                # 更新目标值
                self.target_pos_list_buffer.append(Target_Pos_list) # 目标缓存
                self.Compute_YAW() # 计算目标 YAW

    ############################################# 计算目标 YAW #############################################
    def Compute_YAW(self):
        Vector_3D = self.Drone_target_vec
        self.Drone_target_YAW = np.arctan2(Vector_3D[1], Vector_3D[0])  # 计算 YAW 角度



# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt,           # dt
                ):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    # ---- YOUR CODE HERE ----

    global Drone_Controller


    #0000FF 原始控制命令
    control_command = [sensor_data['x_global'], 
                       sensor_data['y_global'], 
                       1.0,   # 1.0
                       sensor_data['yaw']]

    #0000FF 当前控制命令
    # 判断是否第一次运行
    if Drone_Controller is None:
        Drone_Controller = Class_Drone_Controller(camera_data)  # 创建无人机控制器对象
        print("Drone_Controller Created")

    # 无人机状态更新
    Drone_Controller.update(sensor_data, camera_data) 
    # Drone_Controller.Compute_Target_With_Buffer()
    Drone_Controller.Compute_Target_List_with_Buffer() # 测试 list 更新

    
    # #FF0000 目标测试
    # # 说明：这里发送什么命令无人机就会到什么地方
    # if Drone_Controller.target_buffer:
    #     TARGET = Drone_Controller.target_buffer[-1]
    #     control_command = [TARGET[0] + 0.2, # 这里需要修改，需要视觉计算粉色矩形的法向量
    #                        TARGET[1] - 0.2,
    #                        TARGET[2],
    #                        Drone_Controller.Drone_target_YAW]  # 目标 YAW

    #FF0000 多目标测试
    # 说明：这里发送什么命令无人机就会到什么地方
    if Drone_Controller.target_pos_list_buffer:
        TARGET = Drone_Controller.target_pos_list_buffer[-1][4]
        print(TARGET)

        control_command = [TARGET[0] + 0.2, # 这里需要修改，需要视觉计算粉色矩形的法向量
                           TARGET[1] - 0.2,
                           TARGET[2],
                           Drone_Controller.Drone_target_YAW]  # 目标 YAW
    
    return control_command 

# Ordered as array with: [pos_x_cmd, 
#                         pos_y_cmd, 
#                         pos_z_cmd, 
#                         yaw_cmd] in meters and radians