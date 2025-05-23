import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MotionPlanner3D():
    
    #Question: SIMON PID, what is vel_max set for PID? Check should be same here
    def __init__(self, Gate_points, 
                 start_point = [0,0,0.3], 
                 time_gain = 1.5, 
                 speed_limit = 1.4, 
                 DEBUG = 0):
        
        self.DEBUG = DEBUG
        self.Gate_points = Gate_points
        self.speed_limit = speed_limit

        #FF0000 起点
        self.start_point = start_point

        # 构建两圈 way_points
        way_points = [start_point]
        for point in Gate_points:
            way_points.append(point)
        for point in Gate_points:
            way_points.append(point)
        way_points.append(start_point) # 终点
        
        self.waypoints = way_points
        self.original_waypoints = way_points.copy() # 记录原始轨迹点

        # 飞行时间
        self.t_f = len(self.waypoints) * time_gain  # 20 -> 30


        self.trajectory_setpoints = None # 最终连续轨迹
        self.time_setpoints       = None # 轨迹时间 
        obstacles                 = None
        self.delta_t = None

        # 变量
        self.result_traj_vel_max = None
        self.result_traj_acc_max = None

        # 启动函数
        self.init_params(self.waypoints)
        self.run_planner(obstacles, self.waypoints) # 计算所有数据

        # 轨迹可视化
        if self.DEBUG == 1:
            self.plot_continuous(obstacles, self.waypoints, self.trajectory_setpoints)
        elif self.DEBUG == 2:
            self.plot_discrete(obstacles, self.waypoints, self.get_sparse_trajectory(distance=0.2))

        # ---------------------------------------------------------------------------------------------------- ##
    
    
    
    
    #00FF00 #00FF00
    # 根据起点，终点，以及经过点规划轨迹
    def run_planner(self, obs, path_waypoints):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints
         ## DO NOT MODIFY --------------------------------------------------------------------------------------- ##
    
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.trajectory_velocities , self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs, obs, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def init_params(self, path_waypoints):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        #00FF00
        self.disc_steps = 20                 # Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim    = self.speed_limit   # Velocity limit of the drone (m/s)
        self.acc_lim    = 50.0               # Acceleration limit of the drone (m/s²)

        # Determine the number of segments of the path
        self.times = np.linspace(0, self.t_f, len(path_waypoints)) # The time vector at each path waypoint to traverse (Vector of size m) (must be 0 at start)
        self.delta_t = (self.times[1] - self.times[0]) / self.disc_steps

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        
        # TASK: Fill in the constraint factor matrix values where each row corresponds to the positions, velocities, accelerations, snap and jerk here
        # SOLUTION ---------------------------------------------------------------------------------- ## 
        
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1],            # pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], # vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0],      # acc  
            [60*(t**2), 24*t, 6, 0, 0, 0],             # jerk
            [120*t, 24, 0, 0, 0, 0]                    # snap
        ])

        return A_m

    def compute_poly_coefficients(self, path_waypoints):
        
        
        seg_times = np.diff(self.times) # The time taken to complete each path segment
        m = len(path_waypoints)         # Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs


    def poly_setpoint_extraction(self, poly_coeffs, obs, path_waypoints):
        # ——— unchanged setup ———
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        # 原始时间增量
        dt = self.delta_t
        ds_max = self.vel_lim * dt

        # Helper: 在任意时刻 t 返回位置、速度、航向
        def sample_at(t):
            seg_idx = min(max(np.searchsorted(self.times, t) - 1, 0), len(coeff_x)//6 - 1)
            tau = t - self.times[seg_idx]
            A = self.compute_poly_matrix(tau)
            cx = coeff_x[seg_idx*6:(seg_idx+1)*6]
            cy = coeff_y[seg_idx*6:(seg_idx+1)*6]
            cz = coeff_z[seg_idx*6:(seg_idx+1)*6]

            # 位置
            px = A[0].dot(cx)
            py = A[0].dot(cy)
            pz = A[0].dot(cz)
            # 速度
            vx = A[1].dot(cx)
            vy = A[1].dot(cy)
            vz = A[1].dot(cz)
            # 航向
            yaw = np.rad2deg(np.arctan2(vy, vx))
            return np.array([px, py, pz]), np.array([vx, vy, vz]), yaw

        # 初始化
        t_curr = self.times[0]
        pos_prev, vel_prev, yaw_prev = sample_at(t_curr)
        trajectory        = [[*pos_prev, yaw_prev]]
        velocities        = [vel_prev]
        time_pts          = [t_curr]

        # 主循环
        while t_curr < self.times[-1] - 1e-6:
            # 先尝试 dt 步进
            t_next = min(t_curr + dt, self.times[-1])
            pos_next, vel_next, yaw_next = sample_at(t_next)
            dist = np.linalg.norm(pos_next - pos_prev)

            if dist <= ds_max:
                # 全 dt
                t_sample, pos_sample, vel_sample, yaw_sample = \
                    t_next, pos_next, vel_next, yaw_next
            else:
                # 插值到最大步长
                alpha = ds_max / dist
                t_sample = t_curr + (t_next - t_curr) * alpha
                pos_sample, vel_sample, yaw_sample = sample_at(t_sample)

            trajectory.append([*pos_sample, yaw_sample])
            velocities.append(vel_sample)
            time_pts.append(t_sample)

            t_curr, pos_prev = t_sample, pos_sample

        # 转为 numpy
        trajectory_setpoints = np.array(trajectory)        # shape: (N,4)
        trajectory_velocities = np.array(velocities)       # shape: (N,3)
        time_setpoints        = np.array(time_pts)         # 原始不均匀时间

        # 如果需要重新生成均匀时间戳 (可选)
        N = len(time_pts)
        time_setpoints = np.arange(N) * dt

        return trajectory_setpoints, trajectory_velocities, time_setpoints

    # 计算曲率半径
    def compute_curvature_radius(self):
        """
        计算每个 trajectory_setpoints 的曲率半径（倒数曲率），返回 shape=(N,) 的数组。
        """
        if self.trajectory_setpoints is None:
            raise ValueError("trajectory_setpoints is None, 请先规划轨迹。")
        if self.trajectory_velocities is None:
            raise ValueError("trajectory_velocities is None, 请先规划轨迹。")

        # 取轨迹点
        xyz = self.trajectory_setpoints[:, :3]
        N = len(xyz)

        # 计算一阶导数（速度）和二阶导数（加速度）
        v = np.gradient(xyz, axis=0) / self.delta_t
        a = np.gradient(v, axis=0) / self.delta_t

        # 曲率公式：kappa = |v × a| / |v|^3
        cross = np.cross(v, a)
        v_norm = np.linalg.norm(v, axis=1)
        cross_norm = np.linalg.norm(cross, axis=1)
        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.where(v_norm > 1e-6, cross_norm / (v_norm**3), 0.0)
            curvature_radius = np.where(curvature > 1e-6, 1.0 / curvature, np.inf)

        return curvature_radius



    def plot_obstacle(self, ax, x, y, z, dx, dy, dz, color='gray', alpha=0.3):
        """Plot a rectangular cuboid (obstacle) in 3D space."""
        vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])
        
        faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
                [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
                [vertices[j] for j in [0, 3, 7, 4]], [vertices[j] for j in [1, 2, 6, 5]]]
        
        ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha))
    

    def resample_and_replan(self, distance=1.0):
        """
        根据当前 trajectory_setpoints 采样等间距 waypoints，然后重新规划轨迹。
        """
        if self.trajectory_setpoints is None:
            raise ValueError("trajectory_setpoints is None, 请先运行一次规划。")
        
        # 采样等间距的 waypoints
        traj = self.trajectory_setpoints
        xyz = traj[:, :3]
        dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_dist = cum_dist[-1]
        num_points = int(np.floor(total_dist / distance)) + 1
        sample_dists = np.linspace(0, total_dist, num_points)
        sampled_xyz = np.zeros((num_points, 3))
        for i in range(3):
            sampled_xyz[:, i] = np.interp(sample_dists, cum_dist, xyz[:, i])
        # 重新规划
        self.waypoints = sampled_xyz.tolist()
        self.init_params(self.waypoints)
        self.run_planner(None, self.waypoints)


    def plot_continuous(self, 
             obs,                  # 障碍物
             path_waypoints,       # 路径点 - 离散
             trajectory_setpoints  # 轨迹点 - 接近连续
             ):
        # 创建 figure 和 ax
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # （如果有障碍物的话）画障碍物
        if obs is not None:
            for ob in obs:
                self.plot_obstacle(ax, ob[0], ob[1], ob[2], ob[3], ob[4], ob[5])

        # 画最小劲度轨迹
        ax.plot(
            trajectory_setpoints[:,0],
            trajectory_setpoints[:,1],
            trajectory_setpoints[:,2],
            label="Minimum-Jerk Trajectory",
            linewidth=2
        )

        # 画路径点
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(
            waypoints_x, waypoints_y, waypoints_z,
            color='red', marker='o', label="Waypoints"
        )

        # ——在这里加箭头——
        skip = 10        # 每隔多少个点画一个箭头
        arrow_len = 0.2  # 箭头长度
        xs = trajectory_setpoints[:,0]
        ys = trajectory_setpoints[:,1]
        zs = trajectory_setpoints[:,2]
        yaws = np.deg2rad(trajectory_setpoints[:,3])

        for i in range(0, len(xs), skip):
            x, y, z, yaw = xs[i], ys[i], zs[i], yaws[i]
            dx = np.cos(yaw)
            dy = np.sin(yaw)
            dz = 0
            ax.quiver(
                x, y, z,         # 起点
                dx, dy, dz,      # 方向
                length=arrow_len,
                # normalize=True,
                arrow_length_ratio=0.4,  # 头部占 40%
                pivot='tail',            # 箭尾在 (x,y,z)
                linewidth=1,
                color='black'
            )

        # 设置坐标轴范围、标签、图例
        ax.set_xlim(-2.0, +3.0)
        ax.set_ylim(-1.5, +2.0)
        ax.set_zlim(0, 4)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Motion planning trajectories")
        ax.legend()

        # 手动设置坐标轴范围相等
        xyz_limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        xyz_center = np.mean(xyz_limits, axis=1)
        xyz_radius = (xyz_limits[:,1] - xyz_limits[:,0]).max() / 2
        ax.set_xlim3d([xyz_center[0] - xyz_radius, xyz_center[0] + xyz_radius])
        ax.set_ylim3d([xyz_center[1] - xyz_radius, xyz_center[1] + xyz_radius])
        ax.set_zlim3d([xyz_center[2] - xyz_radius, xyz_center[2] + xyz_radius])

        # 俯视视角
        ax.view_init(elev=90, azim=90)

        plt.show()


    def plot_discrete(self, 
            obs,                  # 障碍物
            path_waypoints,       # 路径点 - 离散
            trajectory_setpoints  # 轨迹点 - 采样点
            ):
        """
        画轨迹点，使用颜色渐变表示速度快慢（蓝色为慢，红色为快）。
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 画障碍物
        if obs is not None:
            for ob in obs:
                self.plot_obstacle(ax, ob[0], ob[1], ob[2], ob[3], ob[4], ob[5])

        # 计算速度（基于相邻点的位置差和时间间隔）
        xs = trajectory_setpoints[:,0]
        ys = trajectory_setpoints[:,1]
        zs = trajectory_setpoints[:,2]
        
        # 计算速度 (如果数据点足够)
        if len(trajectory_setpoints) > 1:
            # 计算位置差
            diffs = np.diff(trajectory_setpoints[:, :3], axis=0)
            # 计算欧几里得距离
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            # 假设时间步长固定 (self.delta_t)
            velocities = distances / self.delta_t
            # 对第一个点重复一次速度值，使长度匹配
            velocities = np.append(velocities[0], velocities)
        else:
            velocities = np.array([0])  # 如果只有一个点
        
        # 归一化速度用于着色 (0 到 1 之间)
        norm_velocities = None
        if np.max(velocities) > np.min(velocities):
            norm_velocities = (velocities - np.min(velocities)) / (np.max(velocities) - np.min(velocities))
        else:
            norm_velocities = np.zeros_like(velocities)
        
        # 创建颜色映射，从蓝色（慢）到红色（快）
        cmap = plt.cm.coolwarm
        colors = cmap(norm_velocities)
        
        # 画轨迹点，颜色代表速度
        scatter = ax.scatter(
            xs, ys, zs,
            c=velocities, 
            cmap='coolwarm', 
            marker='.', 
            label="Trajectory Points",
            s=30, # 点大小
            alpha=0.8 # 透明度
        )
        
        # 添加颜色条，显示速度值
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Velocity (m/s)')
        
        # 画路径点
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(
            waypoints_x, waypoints_y, waypoints_z,
            color='black', marker='o', label="Waypoints", s=80
        )

        # 画箭头表示朝向
        skip = min(10, max(1, len(xs) // 20))  # 动态调整箭头数量
        arrow_len = 0.2  # 箭头长度
        
        if len(trajectory_setpoints[0]) > 3:  # 确保有yaw数据
            yaws = np.deg2rad(trajectory_setpoints[:,3])
            for i in range(0, len(xs), skip):
                x, y, z, yaw = xs[i], ys[i], zs[i], yaws[i]
                dx = np.cos(yaw)
                dy = np.sin(yaw)
                dz = 0
                ax.quiver(
                    x, y, z,
                    dx, dy, dz,
                    length=arrow_len,
                    arrow_length_ratio=0.4,
                    pivot='tail',
                    linewidth=1,
                    color='green'  # 使用绿色区分箭头
                )

        # 设置坐标轴范围、标签、图例
        # ax.set_xlim(-2.0, +3.0)
        # ax.set_ylim(-1.5, +2.0)
        # ax.set_zlim(0, 4)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Trajectory with Velocity Gradient")
        ax.legend()

        # 手动设置坐标轴范围相等
        xyz_limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        xyz_center = np.mean(xyz_limits, axis=1)
        xyz_radius = (xyz_limits[:,1] - xyz_limits[:,0]).max() / 2
        ax.set_xlim3d([xyz_center[0] - xyz_radius, xyz_center[0] + xyz_radius])
        ax.set_ylim3d([xyz_center[1] - xyz_radius, xyz_center[1] + xyz_radius])
        ax.set_zlim3d([xyz_center[2] - xyz_radius, xyz_center[2] + xyz_radius])

        # 俯视视角
        ax.view_init(elev=90, azim=90)
        
        # 添加速度统计信息
        if len(velocities) > 0:
            plt.figtext(0.02, 0.02, f"Min Speed: {np.min(velocities):.2f} m/s\n"
                                f"Max Speed: {np.max(velocities):.2f} m/s\n"
                                f"Avg Speed: {np.mean(velocities):.2f} m/s", 
                    bbox=dict(facecolor='white', alpha=0.5))
            

        # 在 plot_discrete 函数中添加这些调试语句
        print(f"Delta_t: {self.delta_t}")
        # print(f"Velocity statistics:")
        # print(f"  Min velocity: {np.min(velocities):.4f} m/s")
        # print(f"  Max velocity: {np.max(velocities):.4f} m/s")
        # print(f"  Mean velocity: {np.mean(velocities):.4f} m/s")
        # print(f"  Velocity range: {np.max(velocities) - np.min(velocities):.4f} m/s")

        plt.show()