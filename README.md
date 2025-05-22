# Aerial Robotics 
Exercises and the major software project using a Crazyflie implemented in Webots for the MICRO-502 Aerial Robotics course.

**Documentation:** https://micro-502.readthedocs.io



https://github.com/user-attachments/assets/74ce3f74-0939-41c1-b5f0-324baa804558


## **软件项目 Software Project**

### **0.代码说明**
- 在 `main.py` 开头修改使用键盘还是自动路径
- 在 `my_assignment.py` 中编写控制代码

### **1.Webots Python Debug 路径补全**
```json
{
    "python.autoComplete.extraPaths": [ 
        "C:\\Program Files\\Webots\\lib\\controller\\python",
    ],
    "python.analysis.extraPaths": [
        "C:\\Program Files\\Webots\\lib\\controller\\python",
    ],
}
```


## **硬件项目 Hardware Project**

### 硬件任务
- [x] 移植 Project 1 路径规划 Class
- [ ] 优化路径间距

### **0.硬件说明**
- 无人机：
  - Craztflie
- 定位系统：
  - 光流计
  - 红外动捕 Lighthouse

```bash
cfclient 
# 启动客户端监测 定位是否漂移
# 内部有 Lighthouse 校准
```

### **1.Python 路径补全**
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

### **2.Python 连接无人机**
直接运行 python 文件即连接无人机  
正确连接将会出现：
```bash
Connecting to radio://0/30/2M/E7E7E7E713
Connected to radio://0/30/2M/E7E7E7E713
[1333220][Stabilizer]: stateEstimate.x: 0.454 stateEstimate.y: -0.852 stateEstimate.z: 0.008 stabilizer.yaw: -0.621 
[1333270][Stabilizer]: stateEstimate.x: 0.454 stateEstimate.y: -0.852 stateEstimate.z: 0.007 stabilizer.yaw: -0.616
```

### **3.Python 无人机控制函数说明**

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
- **2025.5.6 周二**
  - 拿到无人机硬件
  - 配置无人机控制环境
  - 发现光流计在黑色地面起飞会出现严重漂移导致坠机，需要在白色地面起飞
  - 发现无人机第一次连接 Lighthouse动捕 最好校准，4个基站位置变化将导致无人机位置漂移
- **2025.5.9 周五**
  - 无人机连接到 Lighthouse
  - 发现无人机自动数据融合光流计和红外传感器，计算位置数据
  - 发现无人机飞行不稳定时，需要查看 lighthouse 是否在飘
- **2025.5.16 周五**
  - 无人机成功跟随路径飞行
  - 发现轨迹部分点过于密集，存在提速优化空间
- **2025.5.19 周一**
  - 使用基于时间的轨迹
  - 发现稀疏点速度过快，导致跟踪效果下降
- **2025.5.22 周四**
  - 基于时间 GAIN = 1.5 Distance= 0.5 18.92s
  - 基于时间 GAIN = 1.3 Distance= 0.5 17.41s
  - 基于时间 GAIN = 1.0 Distance= 0.5 寄了   疑似转弯处速度过快
  - 基于时间 GAIN = 1.5 Distance= 1.3 寄了
  - 修改均匀采样为限制最大速度采样
  - 基于时间 GAIN = 1.5 Vel_lim = 1.2 22.08s
  - 基于时间 GAIN = 1.5 Vel_lim= 1.4 20s
  - 基于时间 GAIN = 1.5 Vel_lim= 1.6 20s
  - 基于时间 GAIN = 1.2 Vel_lim= 1.4 19.2s (降落时往前，+3s,实际16s)
  - 基于时间 GAIN = 1.2 Vel_lim= 1.4 17.8s (降落时往前，+3s,实际16s)  time_landing = 0.5s
