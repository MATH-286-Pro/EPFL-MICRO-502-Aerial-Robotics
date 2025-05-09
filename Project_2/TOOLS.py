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


# 自动重连，解决需要手动上电问题
def auto_reconnect(cf, uri:str):
    cf.close_link() # 断开链接
    time.sleep(1)   # 等待1秒
    ps = power_switch.PowerSwitch(uri)
    ps.stm_power_cycle()
    ps.close()