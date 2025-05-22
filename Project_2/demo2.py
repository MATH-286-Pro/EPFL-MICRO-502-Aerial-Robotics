# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, logs the Stabilizer
and prints it to the console. 

The Crazyflie is controlled using the commander interface 

Press q to Kill the drone in case of emergency

After 50s the application disconnects and exits.
"""
import logging
import time
from threading import Timer
import threading

import Project_2.tools as tools
import pandas as pd
import numpy as np

from pynput import keyboard # Import the keyboard module for key press detection

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper, power_switch
from Planning.planning import MotionPlanner3D
#0000FF TODO: CHANGE THIS URI TO YOUR CRAZYFLIE & YOUR RADIO CHANNEL
uri = uri_helper.uri_from_env(default='radio://0/30/2M/E7E7E7E713')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


# 定义收集定位函数
def record_position(le,flight_log)->None:
    pos = le.position
    flight_log.append([
        pos.get('stateEstimate.x', 0),
        pos.get('stateEstimate.y', 0),
        pos.get('stateEstimate.z', 0),
        pos.get('stateEstimate.yaw', 0),
    ])

# def get_real_position(le:Crazyflie)->np.ndarray:
#     pos = le.position
#     return np.array([
#         pos.get('stateEstimate.x', 0),
#         pos.get('stateEstimate.y', 0),
#         pos.get('stateEstimate.z', 0),
#         pos.get('stateEstimate.yaw', 0),
#     ])

# # 定义到达位置判断
# def check_at_command(pos_real, pos_command, threshold=0.02):
#     pos_real    = pos_real[0:3]
#     pos_command = pos_command[0:3]

#     diff = np.linalg.norm(pos_real - pos_command)

#     if diff < threshold:  
#         return True
#     else:
#         return False
    


# Define your custom callback function
def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':  # Check if the "space" key is pressed
                print("Emergency stop triggered!")
                cf.commander.send_stop_setpoint()  # Stop the Crazyflie

                df = pd.DataFrame(flight_log, columns=['x', 'y', 'z', 'yaw'])
                df.to_csv('flight_log.csv', index=False)
                print("飞行数据已保存到 flight_log.csv")
                tools.auto_reconnect(cf, uri)

                return False     # Stop the listener
        except AttributeError:
            pass

    # Start listening for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':

    ############## 无人机初始化 ##############
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    le = tools.LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    # Replace the thread creation with the updated function
    emergency_stop_thread = threading.Thread(target=emergency_stop_callback, args=(cf,))
    emergency_stop_thread.start()

    #00FF00 添加测试代码
    flight_log = [] # 用于记录无人机位置


    print("Starting control")
    while le.is_connected:
        time.sleep(0.01)
        
        # 定义单位
        second  = 1  # 秒
        mm = 0.001   # 毫米
        cm = 0.01    # 厘米
        m  = 1       # 米

        ##################################################### 飞行数据 #####################################################

        # 定义飞行参数
        TIME_TAKE_OFF = 1*second
        TIME_LAND     = 1*second
        HOVER_HEIGHT  = 30*cm  
        
        # 定义飞行轨迹
        Trajectory = tools.Trajectory_Class('position_records.csv', HOVER_HEIGHT)
        TARGET_POINTS = Trajectory.return_gate_points_list()
        planner = MotionPlanner3D(Gate_points = TARGET_POINTS)
        planner.resample_and_replan(distance=1.0)     # 重采样轨迹 #FF0000

        ##################################################### 控制部分 #####################################################

        # 起飞
        tools.FLY_or_LAND(cf, 'takeoff', HOVER_HEIGHT, TIME_TAKE_OFF)

        # 巡航
        POS_COMMAND = planner.trajectory_setpoints

        # 开环飞行
        # for index in range(1,len(POS_COMMAND)):
        #     cf.commander.send_position_setpoint(POS_COMMAND[index][0],
        #                                         POS_COMMAND[index][1],
        #                                         POS_COMMAND[index][2],
        #                                         0) # 关闭 Yaw 轴控制
        #     record_position(le, flight_log) #00FF00 记录飞行数据
        #     time.sleep(0.1)
        
        # 基于时间的飞行
        start_time = time.time()
        index = 0
        while index < len(POS_COMMAND):

            current_time = time.time() - start_time

            if time.time() - start_time >= planner.time_setpoints[index]:
                index += 1

            try:
                cf.commander.send_position_setpoint(POS_COMMAND[index][0],
                                                    POS_COMMAND[index][1],
                                                    POS_COMMAND[index][2],
                                                    0)
                record_position(le, flight_log) #00FF00 记录飞行数据
            except IndexError:
                pass
            time.sleep(0.02)

        # 降落
        tools.FLY_or_LAND(cf, 'land', HOVER_HEIGHT, TIME_LAND)
        ##################################################### 控制部分 #####################################################

       # 保存飞行数据到 CSV
        df = pd.DataFrame(flight_log, columns=['x', 'y', 'z', 'yaw'])
        df.to_csv('flight_log.csv', index=False)
        print("飞行数据已保存到 flight_log.csv")

        tools.auto_reconnect(cf, uri)
        
        break