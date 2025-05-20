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

# def plot_multiple_flight_logs(pos_lists, labels=None):
#     """
#     输入多组 pos_list（每组为 get_real_pos_list 返回的数组），绘制多条三维轨迹，不同颜色区分。
#     参数:
#         pos_lists: list of np.ndarray，每个元素为 shape (N,4) 的轨迹点数组
#         labels: 可选，list of str，每条轨迹的标签
#     """
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     for idx, pos_list in enumerate(pos_lists):
#         color = colors[idx % len(colors)]
#         label = labels[idx] if labels and idx < len(labels) else f'Flight {idx+1}'
#         ax.plot(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], label=label, color=color)
#         ax.scatter(pos_list[0, 0], pos_list[0, 1], pos_list[0, 2], color=color, marker='o')     # Start
#         ax.scatter(pos_list[-1, 0], pos_list[-1, 1], pos_list[-1, 2], color=color, marker='x')  # End


#     # 手动设置坐标轴范围相等
#     xyz_limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     xyz_center = np.mean(xyz_limits, axis=1)
#     xyz_radius = (xyz_limits[:,1] - xyz_limits[:,0]).max() / 2
#     ax.set_xlim3d([xyz_center[0] - xyz_radius, xyz_center[0] + xyz_radius])
#     ax.set_ylim3d([xyz_center[1] - xyz_radius, xyz_center[1] + xyz_radius])
#     ax.set_zlim3d([xyz_center[2] - xyz_radius, xyz_center[2] + xyz_radius])


#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Multiple Flight Trajectories')
#     ax.legend()
#     ax.view_init(elev=90, azim=90)
#     plt.show()


def plot_multiple_flight_logs(pos_lists, gata_pos_array, type = "continous"):
    """
    输入多组 pos_list（每组为 get_real_pos_list 返回的数组），绘制多条三维轨迹，不同颜色区分。
    参数:
        pos_lists: list of np.ndarray，每个元素为 shape (N,4) 的轨迹点数组
        labels: 可选，list of str，每条轨迹的标签
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    labels = None

    colors = ['blue', 'orange', 'red', 'purple', 'pink','brown', 'gray', 'olive', 'cyan']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gata_pos_array[:, 0], gata_pos_array[:, 1], gata_pos_array[:, 2], label="Gate", color='green', s=50, alpha=1.0)

    for idx, pos_list in enumerate(pos_lists):
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else f'Flight {idx+1}'

        if type == "continous":
            ax.plot(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], label=label, color=color)
        elif type == "discrete":
            ax.scatter(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], label=label, color=color, s=5)

        # 起点和终点标记
        ax.scatter(pos_list[0, 0], pos_list[0, 1], pos_list[0, 2], color=color, marker='o')     # Start
        ax.scatter(pos_list[-1, 0], pos_list[-1, 1], pos_list[-1, 2], color=color, marker='x')  # End
    
    # 手动设置坐标轴范围相等
    xyz_limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    xyz_center = np.mean(xyz_limits, axis=1)
    xyz_radius = (xyz_limits[:,1] - xyz_limits[:,0]).max() / 2
    ax.set_xlim3d([xyz_center[0] - xyz_radius, xyz_center[0] + xyz_radius])
    ax.set_ylim3d([xyz_center[1] - xyz_radius, xyz_center[1] + xyz_radius])
    ax.set_zlim3d([xyz_center[2] - xyz_radius, xyz_center[2] + xyz_radius])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiple Flight Trajectories')
    ax.legend()
    ax.view_init(elev=90, azim=90)
    plt.show()