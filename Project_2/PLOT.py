import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_real_pos_list(csv_path = "flight_log.csv"):
    data = pd.read_csv(csv_path)

    n = len(data['x'])

    x_array   = data['x'].to_numpy()
    y_array   = data['y'].to_numpy()
    z_array   = data['z'].to_numpy()
    yaw_array = data['yaw'].to_numpy()

    pos_list = np.array([x_array, y_array, z_array, yaw_array]).T

    return pos_list


def plot_flight_log(pos_list):
    """
    接收 get_real_pos_list 返回的 pos_list 并绘制三维轨迹
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], label='Flight Path', color='blue')
    ax.scatter(pos_list[0, 0], pos_list[0, 1], pos_list[0, 2], color='green', label='Start')
    ax.scatter(pos_list[-1, 0], pos_list[-1, 1], pos_list[-1, 2], color='red', label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Flight Trajectory')
    ax.legend()
    plt.show()


def plot_multiple_flight_logs(pos_lists, labels=None):
    """
    输入多组 pos_list（每组为 get_real_pos_list 返回的数组），绘制多条三维轨迹，不同颜色区分。
    参数:
        pos_lists: list of np.ndarray，每个元素为 shape (N,4) 的轨迹点数组
        labels: 可选，list of str，每条轨迹的标签
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for idx, pos_list in enumerate(pos_lists):
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else f'Flight {idx+1}'
        ax.plot(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], label=label, color=color)
        ax.scatter(pos_list[0, 0], pos_list[0, 1], pos_list[0, 2], color=color, marker='o')  # Start
        ax.scatter(pos_list[-1, 0], pos_list[-1, 1], pos_list[-1, 2], color=color, marker='x')  # End

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiple Flight Trajectories')
    ax.legend()
    plt.show()