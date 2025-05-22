# import tools
# from Planing.planning import MotionPlanner3D

# Trajectory = tools.Trajectory_Class('position_records.csv')

# planner = MotionPlanner3D(waypoints  = Trajectory.point_list,
#                           DEBUG = True)


import os
import sys
import numpy as np
import pandas as pd

# 将上一级目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import tools
import PLOT
from Planning.planning import MotionPlanner3D


cm = 0.01
Trajectory = tools.Trajectory_Class('position_records.csv') # 仅返回门的位置
test_planner = MotionPlanner3D(Gate_points = Trajectory.return_gate_points_list(),DEBUG = 0)

#FF0000 测试重采样避免高速轨迹
test_planner.resample_and_replan(distance=1.0)

# data_OpenLoop = PLOT.get_real_pos_list("flight_log_10_OL.csv")
data_CloseLoop = PLOT.get_real_pos_list("flight_log_14_TimeBased.csv")
PLOT.plot_multiple_flight_logs([
                                # data_OpenLoop, 
                                data_CloseLoop, 
                                test_planner.trajectory_setpoints,
                                np.array(test_planner.waypoints),
                                ],
                                np.array(test_planner.Gate_points), # 标记门位置
                                type='discrete'                     # 绘制离散轨迹
                                )

