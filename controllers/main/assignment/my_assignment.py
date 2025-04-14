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


# 目标流程：
#  第一圈
#   1.确定目标位置 (使用视觉测定 gate 中心位置)
#  
#  第二圈
#   1.进行最优路径规划

# 视觉处理步骤
# 1.滤波
# 2.相机参数标定


# 宏定义
W = 0  # 四元数下标
X = 1
Y = 2
Z = 3
f_pixel = 161.013922282   # 相机焦距
vector_from_drone_to_cam = np.array([0.03,0,00,0.01]) # 无人机中心到相机偏移向量

########################################## 基础函数 ##########################################
def ROUND(a):
    return (a*100).astype(int)/100

# 基于点斜式的对角线交点计算
# def compute_center(rect):

#     p0 = rect[0,0]; p1 = rect[1,0]; p2 = rect[2,0]; p3 = rect[3,0]

#     x0 = p0[0]; y0 = p0[1]
#     x1 = p1[0]; y1 = p1[1]
#     x2 = p2[0]; y2 = p2[1]
#     x3 = p3[0]; y3 = p3[1]

#     k0 = (y2-y0)/(x2-x0)
#     k1 = (y3-y1)/(x3-x1)

#     xc = (k1*x1-k0*x0-(y1-y0))/(k1-k0)
#     yc = k1*(xc-x1)+y1

#     return xc, yc

# 中心点基础函数
def compute_center(rect, eps=1e-6):

    # 取出四个点
    p0 = rect[0, 0]
    p1 = rect[1, 0]
    p2 = rect[2, 0]
    p3 = rect[3, 0]
    
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

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


# 四元数基础函数
def quat_mutiplication(q1,q2):
    ans = np.array([q1[W]*q2[W] - q1[X]*q2[X] - q1[Y]*q2[Y] - q1[Z]*q2[Z],
                    q1[W]*q2[W] + q1[X]*q2[X] + q1[Y]*q2[Y] - q1[Z]*q2[Z],
                    q1[W]*q2[W] - q1[X]*q2[X] + q1[Y]*q2[Y] + q1[Z]*q2[Z],
                    q1[W]*q2[W] + q1[X]*q2[X] - q1[Y]*q2[Y] + q1[Z]*q2[Z],])
    return ans

def quat_rotate(P1, Q):
    Q_prim = np.array([Q[W], -Q[X], -Q[Y], -Q[Z]])
    P2 = quat_mutiplication(quat_mutiplication(Q,P1),Q_prim)
    return P2

def vector_rotate(p1, Q):
    P1 = np.array([0, p1[0], p1[1], p1[2]])
    P2 = quat_rotate(P1, Q)
    return P2[1:4]  # 返回旋转后的向量部分


def get_quat_from_sensor(sensor_data):
    q_x = sensor_data['q_x']
    q_y = sensor_data['q_y']
    q_z = sensor_data['q_z']
    q_w = sensor_data['q_w']

    # 将四元数转换为 numpy 数组
    quat = np.array([q_w, q_x, q_y, q_z])
    
    return quat

########################################## 基础函数 ##########################################


# 图像处理函数
def img_debug(camera_data,
              sensor_data):

    # 原始 BGR 图像（忽略 Alpha）
    b, g, r, a = cv2.split(camera_data)
    bgr_image  = cv2.merge([b, g, r])
    # cv2.imshow("Original BGR Image", bgr_image)

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
    cv2.imshow("Gray", binary_mask)                              # 灰度图 / mask
    pink_only = cv2.bitwise_and(bgr, bgr, mask=binary_mask)      # 应用 mask 扣图
    cv2.imshow("Pink", pink_only)                                # 粉色图

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
    if largest_rect is not None:
        
        # 计算中心点
        rect_center_x, rect_center_y = compute_center(largest_rect) # np.Float
        cX_pixel = int(rect_center_x)                               # int
        cY_pixel = int(rect_center_y)                               # int

        # 在原图上画出四个点
        for point in largest_rect:
            x, y = point[0]
            cv2.circle(bgr_image, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(bgr_image, (cX_pixel, cY_pixel), 5, (0, 255, 0), -1)
        cv2.imshow("Rectangle Corners", bgr_image)

        # #00FF00 
        # 计算目标点位置
        cam_delta_x = rect_center_x - cam_center_x
        cam_delta_y = rect_center_y - cam_center_y

        # print('center %.2f, %.2f' % (cam_delta_x, cam_delta_y))

        # 无人机坐标系：目标方向向量
        Vector_Direct_Cam2Target_DroneFrame = np.array([f_pixel, -cam_delta_x, -cam_delta_y])

    else:
        # 将不会显示四个点
        cv2.imshow("Rectangle Corners", bgr_image)

    # 坐标变换
    Q         = get_quat_from_sensor(sensor_data)  
    Q_reverse = np.array([Q[W], -Q[X], -Q[Y], -Q[Z]])
    Vector_Direct_Cam2Target_WolrdFrame = vector_rotate(Vector_Direct_Cam2Target_DroneFrame, Q_reverse)  # 旋转向量

    print(Vector_Direct_Cam2Target_DroneFrame)
    # print(Vector_Direct_Cam2Target_WolrdFrame)

    cv2.waitKey(1)



########################################## 四元数部分 ######################################################
def quat_debug(sensor_data_copy):

    q_x = sensor_data_copy['q_x']
    q_y = sensor_data_copy['q_y']
    q_z = sensor_data_copy['q_z']
    q_w = sensor_data_copy['q_w']

    x_global = sensor_data_copy['x_global']
    y_global = sensor_data_copy['y_global']
    z_global = sensor_data_copy['z_global']

    P_global = np.array([0,x_global,y_global,z_global])
    
    q = np.array([q_w, q_x, q_y, q_z])

    P_local = quat_rotate(P_global, q)

    # print('local = ', ROUND(P_local[X:Z+1]), 'global = ', ROUND(P_global[X:Z+1]))

    return 0
########################################## 四元数部分 ######################################################



# 无人机控制函数
def get_command(sensor_data,  # 传感器数据 (详见上面的信息)
                camera_data,  # 相机数据
                dt            # dt
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


    #0000FF


    #0000FF 当前控制命令
    # 说明：这里发送什么命令无人机就会到什么地方

    control_command = [sensor_data['x_global'], 
                       sensor_data['y_global'], 
                       1.0,   # 1.0
                       sensor_data['yaw']]
    
    return control_command 

# Ordered as array with: [pos_x_cmd, 
#                         pos_y_cmd, 
#                         pos_z_cmd, 
#                         yaw_cmd] in meters and radians