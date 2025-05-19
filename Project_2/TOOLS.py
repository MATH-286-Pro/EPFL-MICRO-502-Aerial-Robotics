# 自定义工具

import logging
import time
from threading import Timer
import threading

from pynput import keyboard # Import the keyboard module for key press detection

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper, power_switch

import numpy as np
import pandas as pd




class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 10s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # User Define Variable 用户自定义数据
        self.position = {}   # 位置

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')

        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 50s     #0000FF TODO: CHANGE THIS TO YOUR NEEDS
        t = Timer(50, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""

        # # Print the data to the console
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()
        #0000FF 数据记录
        for name, value in data.items():
            self.position[name] = value
        
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in self.position.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

# 数据读取类
class Trajectory_Class:

    def __init__(self, 
                 path: str,
                 Hover_Height: float = 0.30):
        
        self.file_path = path
        self.full_data = None
        self.Hover_Height = Hover_Height
        self.point_list = []

        # 启动函数
        self.read_data()
        self.process_data()
        self.check_and_fix_trajectory()
    
    def read_data(self):
        self.full_data = pd.read_csv(self.file_path)
    
    def process_data(self):
        x = self.full_data['avg_x'].values
        y = self.full_data['avg_y'].values
        z = self.full_data['avg_z'].values

        ######### 加入绕两圈需要的数据点 #########
        for index in range(len(x)):
            point = [x[index], y[index], z[index]]
            self.point_list.append(point)

        for index in range(len(x)):
            point = [x[index], y[index], z[index]]
            self.point_list.append(point)
    
    def check_and_fix_trajectory(self):
        point_start = np.array(self.point_list[0])
        point_end   = np.array(self.point_list[-1])

        # 如果记录了起点 修正悬停点
        if np.linalg.norm(point_start - np.array([0,0,0])) < 0.5:
            self.point_list[0] = np.array([0.0, 0.0, self.Hover_Height])

        # 如果没记录起点 加入悬停点
        else:
            self.point_list.insert(0, [0.0, 0.0, self.Hover_Height])
        
        # 如果记录了终点 修正悬停点
        if np.linalg.norm(point_start - np.array([0,0,0])) < 0.5:
            self.point_list[-1] = np.array([0.0, 0.0, self.Hover_Height])
        
        # 如果没记录终点 加入悬停点
        else:
            self.point_list.append([0.0, 0.0, self.Hover_Height])






# 封装原地起飞/降落指令
def FLY_or_LAND(cf: Crazyflie, 
                type:str, 
                HEIGHT, 
                TIME):

    if type == 'takeoff':
        for y in range(TIME):
            cf.commander.send_hover_setpoint(0,0,0,y/TIME * HEIGHT)
            time.sleep(0.1)
        for _ in range(10):
            cf.commander.send_hover_setpoint(0,0,0,HEIGHT)
            time.sleep(0.1)
    
    elif type == 'land':
        for _ in range(TIME):
            cf.commander.send_hover_setpoint(0,0,0,HEIGHT)
            time.sleep(0.1)
        for y in range(TIME):
            cf.commander.send_hover_setpoint(0,0,0, (TIME - y)/TIME * HEIGHT)
            time.sleep(0.1)
        cf.commander.send_stop_setpoint()

def position_smooth_change(cf: Crazyflie,
                           start_pos:list,
                           end_pos:  list,
                           TIME:     int,):
    
    start_pos = np.array(start_pos)
    end_pos   = np.array(end_pos)
    pos_smooth_list = np.linspace(start_pos, end_pos, TIME)
    
    for i in range(TIME):
        cf.commander.send_position_setpoint(pos_smooth_list[i][0],
                                         pos_smooth_list[i][1],
                                         pos_smooth_list[i][2],
                                         pos_smooth_list[i][3])
        time.sleep(0.1)
    

    


# 自动重连，解决需要手动上电问题
def auto_reconnect(cf, uri:str):
    cf.close_link() # 断开链接
    time.sleep(1)   # 等待1秒
    ps = power_switch.PowerSwitch(uri)
    ps.stm_power_cycle()
    ps.close()