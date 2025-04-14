# Aerial Robotics 
Exercises and the major software project using a Crazyflie implemented in Webots for the MICRO-502 Aerial Robotics course.

**Documentation:** https://micro-502.readthedocs.io


## 说明
- 在 `main.py` 开头修改使用键盘还是自动路径
- 在 `my_assignment.py` 中编写控制代码


## 键盘控制命令
```python
if key == ord('W'):
    forward_velocity = 2.0
elif key == ord('S'):
    forward_velocity = -2.0
elif key == ord('A'):
    left_velocity = 2.0
elif key == ord('D'):
    left_velocity = -2.0
elif key == ord('Q'):
    yaw_rate = 1.0
elif key == ord('E'):
    yaw_rate = -1.0
elif key == ord('X'):
    altitude_velocity = 0.3
elif key == ord('Z'):
    altitude_velocity = -0.3
key = self.keyboard.getKey()
```


