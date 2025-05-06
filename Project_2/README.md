# 硬件项目说明

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