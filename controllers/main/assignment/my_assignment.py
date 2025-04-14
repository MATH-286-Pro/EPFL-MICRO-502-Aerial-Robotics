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
W = 0  # 四元数下标
X = 1
Y = 2
Z = 3

# 单位化函数
def Unit_Vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # 返回原始向量
    else:
        return v / norm  # 返回单位化向量


# 四元数基础函数
def quat_mutiplication(q1,q2):
    ans = np.array([q1[W]*q2[W] - q1[X]*q2[X] - q1[Y]*q2[Y] - q1[Z]*q2[Z],
                    q1[W]*q2[X] + q1[X]*q2[W] + q1[Y]*q2[Z] - q1[Z]*q2[Y],
                    q1[W]*q2[Y] - q1[X]*q2[Z] + q1[Y]*q2[W] + q1[Z]*q2[X],
                    q1[W]*q2[Z] + q1[X]*q2[Y] - q1[Y]*q2[X] + q1[Z]*q2[W],])
    return ans

def quat_rotate(P1, Q):
    Q_prim = np.array([Q[W], -Q[X], -Q[Y], -Q[Z]])
    P2 = quat_mutiplication(quat_mutiplication(Q,P1),Q_prim)
    return P2

def vector_rotate(p1, Q):
    P1 = np.array([0, p1[0], p1[1], p1[2]])
    P2 = quat_rotate(P1, Q)
    return P2[1:4]  # 返回旋转后的向量部分


# 目标中心点基础函数
def compute_target_center(rect, eps=1e-6):

    # 取出四个点
    x0, y0 = rect[0, 0]
    x1, y1 = rect[1, 0]
    x2, y2 = rect[2, 0]
    x3, y3 = rect[3, 0]

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

    return x, y


# 定义无人机类
class Class_Drone_Controller:

    def __init__(self):

        # 基本参数
        self.f_pixel = 161.013922282   # 相机焦距
        self.vector_Drone2Cam_DroneFrame = np.array([0.03,0,00,0.01]) # 无人机中心到相机偏移向量
        
        # 全局变量 (三角定位)
        self.position_buffer = [] # 位置缓存
        self.vector_buffer   = [] # 方向缓存
        self.min_cumulative_baseline = 0.3  # 设定累计基线距离阈值

        self.target_buffer   = [] # 目标缓存

        # 无人机信息
        self.sensor_data    = None  # 无人机传感器数据
        self.camera_data    = None  # 相机数据

        # 计算数据
        self.Drone_Pos_Global = None  
        self.Drone_target_vec = None  


    ########################################## 更新函数 ##########################################

    # 更新无人机位置 + 更新相机数据
    def update(self, sensor_data, camera_data):
        
        # 更新数据 + 相机
        self.sensor_data  = sensor_data
        self.camera_data  = camera_data

        # 更新位置 + 相机目标
        self.Drone_Pos_Global = self.get_position_global()  # 无人机全局坐标系下位置
        self.Drone_target_vec = self.img_to_vector()        # 相机坐标系下目标位置

    ########################################## 传感器函数 ##########################################
    def get_quat_from_sensor(self):
        q_x = self.sensor_data['q_x']
        q_y = self.sensor_data['q_y']
        q_z = self.sensor_data['q_z']
        q_w = self.sensor_data['q_w']
        quat = np.array([q_w, q_x, q_y, q_z])
        return quat

    def get_position_global(self):
        x_global = self.sensor_data['x_global']
        y_global = self.sensor_data['y_global']
        z_global = self.sensor_data['z_global']
        position = np.array([x_global, y_global, z_global])
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


    ########################################## 图像处理函数 ##########################################
    def img_to_vector(self):

        # 原始 BGR 图像（忽略 Alpha）
        b, g, r, a = cv2.split(self.camera_data)
        bgr_image  = cv2.merge([b, g, r])

        # 获取图像中心位置
        cam_center_y, cam_center_x, _ = bgr_image.shape
        cam_center_x /= 2
        cam_center_y /= 2

        # 直接用 RGB 图像扣图
        bgr = bgr_image.copy()

        # OpenCV 是 BGR 顺序！
        upper_pink  = np.array([255, 185, 255])  # B, G, R
        lower_pink  = np.array([190, 60, 190])   
        binary_mask = cv2.inRange(bgr, lower_pink, upper_pink)

        # 可视化 mask 和提取结果
        # cv2.imshow("Gray", binary_mask)                              # 灰度图 / mask
        # pink_only = cv2.bitwise_and(bgr, bgr, mask=binary_mask)      # 应用 mask 扣图
        # cv2.imshow("Pink", pink_only)                                # 粉色图

        # 2.轮廓提取
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3.轮廓近似
        # 遍历所有轮廓，找到拟合为四边形且面积最大的那个
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

        # 4. 可视化
        # 看见方框
        cv2.waitKey(1) # 如果放在 return 后面会报错

        if largest_rect is not None:
            
            # 计算中心点
            rect_center_x, rect_center_y = compute_target_center(largest_rect) # np.Float64

            # 在原图上画出四个点
            for point in largest_rect:
                x, y = point[0]
                cv2.circle(bgr_image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(bgr_image, (int(rect_center_x), int(rect_center_y)), 5, (0, 255, 0), -1)
            cv2.imshow("Rectangle Corners", bgr_image)

            # 目标位置：减掉相机中心位置
            cam_delta_x = rect_center_x - cam_center_x
            cam_delta_y = rect_center_y - cam_center_y

            # 目标方向：无人机坐标系
            Vector_Cam2Target_DroneFrame = np.array([self.f_pixel, -cam_delta_x, -cam_delta_y])

            # 目标方向：无人机坐标系 -> 世界坐标系
            Vector_Cam2Target_WorldFrame = self.Convert_Frame_Drone2World(Vector_Cam2Target_DroneFrame)
            Vector_Cam2Target_WorldFrame = Unit_Vector(Vector_Cam2Target_WorldFrame)  # 单位化

            return Vector_Cam2Target_WorldFrame 

        # 没看见方框  将不会显示四个点
        else:
            cv2.imshow("Rectangle Corners", bgr_image)
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

        print("目标全局坐标：",T)

        return T

        # 点1 = [2.12, 1.84, 1.24]
        # 点2 = [5.12, 2.30, 0.78]


    ############################################# 三角定位，缓存更新 #############################################
    def Compute_Target_With_Buffer(self):

        # 更新位置 + 视觉
        # 如果视野内无目标，不能将 None 添加到缓存中
        if (self.Drone_Pos_Global is not None) and (self.Drone_target_vec is not None):
            self.position_buffer.append(self.Drone_Pos_Global)
            self.vector_buffer.append(self.Drone_target_vec)

        # 计算累计位移
        if len(self.position_buffer) >= 2: # 至少两帧数据
            cumulative_baseline = np.linalg.norm(self.position_buffer[-1] - self.position_buffer[0])

            if cumulative_baseline >= self.min_cumulative_baseline:
                T = self.target_positioning(self.position_buffer[-1], 
                                            self.position_buffer[0],
                                            self.vector_buffer[-1],
                                            self.vector_buffer[0])
                # 清空缓存
                self.position_buffer = []
                self.vector_buffer   = []

                self.target_buffer.append(T) # 目标缓存

                print(T)

                return T
        
        return None
    ############################################# 三角定位，缓存更新 #############################################





# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt,           # dt
                Drone_Controller # 无人机控制器类
                ):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    # ---- YOUR CODE HERE ----

    #0000FF 调试区域
    # print(sensor_data)

    #0000FF 当前控制命令
    # 说明：这里发送什么命令无人机就会到什么地方

    control_command = [sensor_data['x_global'], 
                       sensor_data['y_global'], 
                       1.0,   # 1.0
                       sensor_data['yaw']]
    
    #FF0000 目标测试
    if Drone_Controller.target_buffer[-1] is not None:
        TARGET = Drone_Controller.target_buffer[-1]
    control_command = [TARGET[0],
                       TARGET[1],
                       TARGET[2],
                       sensor_data['yaw']]

    
    return control_command 

# Ordered as array with: [pos_x_cmd, 
#                         pos_y_cmd, 
#                         pos_z_cmd, 
#                         yaw_cmd] in meters and radians