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



# 图像处理函数
def img_debug(camera_data):

    # 假设 camera_data 是 (H, W, 4)，RGBA
    b, g, r, a = cv2.split(camera_data)

    # 原始 BGR 图像（忽略 Alpha）
    bgr_image = cv2.merge([b, g, r])
    cv2.imshow("Original BGR Image", bgr_image)


    # 直接用 RGB 图像扣图
    bgr = bgr_image.copy()

    # OpenCV 是 BGR 顺序！
    upper_pink = np.array([255, 185, 255])  # B, G, R
    lower_pink = np.array([190, 60, 190])   
    mask       = cv2.inRange(bgr, lower_pink, upper_pink)

    # 可视化 mask 和提取结果
    cv2.imshow("Pink Mask", mask)

    # 应用 mask 扣图
    pink_only = cv2.bitwise_and(bgr, bgr, mask=mask)
    cv2.imshow("Pink Extracted", pink_only)
    cv2.waitKey(1)




def img_process(camera_data):



    pass




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