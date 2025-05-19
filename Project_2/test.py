# import TOOLS
# from Planing.planning import MotionPlanner3D

# Trajectory = TOOLS.Trajectory_Class('position_records.csv')

# planner = MotionPlanner3D(waypoints  = Trajectory.point_list,
#                           DEBUG = True)


import os
import sys

# 将上一级目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import TOOLS
import pandas as pd
import PLOT

from Planing.planning import MotionPlanner3D

cm = 0.01
HOVER_HEIGHT  = 60*cm  
Trajectory = TOOLS.Trajectory_Class('position_records.csv',
                                    Hover_Height = HOVER_HEIGHT)
test_planner = MotionPlanner3D(waypoints = Trajectory.point_list,
                               DEBUG = 0)

# data_OpenLoop = PLOT.get_real_pos_list("flight_log_10_OL.csv")
data_CloseLoop = PLOT.get_real_pos_list("flight_log_14_TimeBased.csv")
PLOT.plot_multiple_flight_logs([
                                # data_OpenLoop, 
                                data_CloseLoop, 
                                test_planner.trajectory_setpoints])
