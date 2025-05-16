# Aerial Robotics 
Exercises and the major software project using a Crazyflie implemented in Webots for the MICRO-502 Aerial Robotics course.

**Documentation:** https://micro-502.readthedocs.io



https://github.com/user-attachments/assets/74ce3f74-0939-41c1-b5f0-324baa804558


## **软件项目 Software Project**

### 说明
- 在 `main.py` 开头修改使用键盘还是自动路径
- 在 `my_assignment.py` 中编写控制代码


### 任务
- [ ] 调整 PID，使基于时间的轨迹规划减小稳态误差



## **硬件项目 Hardware Project**

## Python 路径补全
```json
{
    "python.autoComplete.extraPaths": [ 
        "${workspaceFolder}\\crazyflie-lib-python"
    ],
    "python.analysis.extraPaths": [
        "${workspaceFolder}\\crazyflie-lib-python"
    ],
}
```

## Python 连接无人机
直接运行 python 文件即连接无人机  
正确连接将会出现：
```bash
Connecting to radio://0/30/2M/E7E7E7E713
Connected to radio://0/30/2M/E7E7E7E713
[1333220][Stabilizer]: stateEstimate.x: 0.454 stateEstimate.y: -0.852 stateEstimate.z: 0.008 stabilizer.yaw: -0.621 
[1333270][Stabilizer]: stateEstimate.x: 0.454 stateEstimate.y: -0.852 stateEstimate.z: 0.007 stabilizer.yaw: -0.616
```

## Python 无人机控制函数说明

```python
# crayflie-lib-python/commander.py 文件说明

# 盘旋命令 控制 vx vy yawrate z
def send_hover_setpoint(self, vx, vy, yawrate, zdistance):

# 位置控制命令 ⭐
def send_position_setpoint(self, x, y, z, yaw):

# 完全控制命令 
def send_full_state_setpoint(self, pos, vel, acc, orientation, rollrate, pitchrate, yawrate):

```


## **项目日志 Project Log**
- **2025.4.27 周日**
  - 删除无用变量与函数
  - 修改正弦扫描模式为匀速扫描
- **2025.5.9 周五**
  - 无人机连接到 Lighthouse
  - 发现无人机自动数据融合光流计和红外传感器，计算位置数据
  - 发现无人机飞行不稳定时，需要查看 lighthouse 是否在飘

