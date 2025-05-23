import numpy as np
from collections import defaultdict

class AggregatedExtractor:
    def __init__(self, data_list, gate_point=(4,4,4), dist_thresh=0.5,
                 angle_range=(45,135), cluster_dist=0.8):
        self.gate_center = np.array(gate_point, float)
        self.T, self.min_ang, self.max_ang = dist_thresh, *angle_range
        self.cluster_dist = cluster_dist

        # 输入数据：每个元素为 shape (5,3) 的 np.ndarray，分别对应 P0..P4
        self.data_list = data_list
        # 提取所有 P4 作为中心点序列
        self.points4 = np.array([d[4] for d in data_list], float)

        # 自建数据存储
        self.P_A_aggregated                = None  # 聚合后点 + 方向
        self.G_P_A_aggregated_sorted       = None  # 排序后的聚合结果
        self.G_P_A_aggregated_sorted_shift = None  # 平移调整后的坐标

        # 生成扇区划分，并立即执行聚合流程
        self.generate_sector_angles()
        self.compute_conditional_idxs()
        self.sort_aggregated()           # 聚类排序

    def compute_mask(self):
        # 判断相邻 P4 点位移动是否小于阈值
        dists = np.linalg.norm(np.diff(self.points4, axis=0), axis=1)
        mask = np.concatenate(([True], dists < self.T))
        return mask

    # 判断角度 + 叉乘，计算索引并聚合每个簇的中心点与平均方向
    # self.data_filtered_sorted =  Gate_Point 和 Gate_Arrow
    def compute_conditional_idxs(self):
        mask = self.compute_mask()
        idxs = []
        arrow_dict = {}
        # 筛选满足距离和角度条件的索引，并计算箭头方向
        for i, pt4 in enumerate(self.points4):
            if not mask[i]:
                continue
            v1 = pt4 - self.gate_center
            # P0 和 P3
            p0, p3 = self.data_list[i][0], self.data_list[i][3]
            # 计算摄像机光轴方向
            theta = np.arctan2(*(p0[:2] - p3[:2])[::-1]) - np.pi/2
            v2 = np.array([np.cos(theta), np.sin(theta), 0.])
            # 计算夹角
            cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
            ang = np.degrees(np.arccos(cosang))
            if self.min_ang <= ang <= self.max_ang:
                idxs.append(i)
                # 如果 v1 与 v2 的叉乘在 z 轴为负，则反转方向
                cross_z = np.cross(v1, v2)[2]
                arrow_dict[i] = -v2 if cross_z < 0 else v2

        # 基于空间距离进行聚类
        clusters, curr = [], [idxs[0]] if idxs else []
        for a, b in zip(idxs, idxs[1:]):
            if np.linalg.norm(self.points4[b] - self.points4[a]) < self.cluster_dist:
                curr.append(b)
            else:
                clusters.append(curr)
                curr = [b]
        if curr:
            clusters.append(curr)

        # 聚合每个簇的中心点与平均箭头方向
        agg = []
        for cl in clusters:
            pts = self.points4[cl]
            center = pts.mean(axis=0)
            arrows = np.array([arrow_dict[i] for i in cl])
            avg_arrow = arrows.mean(axis=0)
            agg.append({'Point': center, 'Arrow': avg_arrow})

        self.P_A_aggregated = agg
        return agg

    ############################################## 扇区划分与排序 #######################################
    def generate_sector_angles(self):
        gate, none = np.deg2rad(35), np.deg2rad(25)
        base = -np.pi - gate/2 + gate + none
        self.sectors = [(base + i*(gate + none), base + i*(gate + none) + gate) for i in range(5)]

    def _sector(self, pt2d):
        ang = np.arctan2(*(pt2d - self.gate_center[:2])[::-1])
        for idx, (s, e) in enumerate(self.sectors):
            if s <= ang <= e:
                return f'Gate{idx}', idx
        return None, None

    def sort_aggregated(self):
        groups = defaultdict(list)
        for item in self.P_A_aggregated:
            label, idx = self._sector(item['Point'][:2])
            if idx is not None:
                groups[idx].append((item['Point'], item['Arrow']))

        result = []
        for idx in sorted(groups):
            pts, arrs = zip(*groups[idx])
            mean_pt = np.mean(pts, axis=0)
            mean_arr = np.mean(arrs, axis=0)
            result.append((f'Gate{idx}', mean_pt, mean_arr))

        self.G_P_A_aggregated_sorted = result

    def convert_to_planning(self):
        # 返回 [(x,y,z), ...]
        return [tuple(np.round(pt, 3)) for _, pt, _ in self.G_P_A_aggregated_sorted]

    def convert_to_planning_shift(self, shift = 0.3):
        self.G_P_A_aggregated_sorted_shift = []
        for label, center, arrow in self.G_P_A_aggregated_sorted:
            angle = np.arctan2(arrow[1], arrow[0])
            new_dir = angle - np.pi/2
            new_vec = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
            new_pt = center + shift * new_vec
            point = tuple(np.round(new_pt, 3))
            self.G_P_A_aggregated_sorted_shift.append((label, point, arrow))
        return [pt for _, pt, _ in self.G_P_A_aggregated_sorted_shift]

    def convert_to_planning_shift_time_customized(self, shift = 0.3):
        self.G_P_A_aggregated_sorted_shift = []
        for label, center, arrow in self.G_P_A_aggregated_sorted:
            angle = np.arctan2(arrow[1], arrow[0])
            new_dir = angle - np.pi/2
            new_vec = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
            new_pt = center + shift * new_vec
            new_pt[2] -= 0.2 # 基于时间导航需要降低高度
            point = tuple(np.round(new_pt, 3))
            self.G_P_A_aggregated_sorted_shift.append((label, point, arrow))
        return [pt for _, pt, _ in self.G_P_A_aggregated_sorted_shift]
    

    
    #FF0000
    def shift_points_bidirectional(self, distance):

        shifted_points = []
        for label, pt, arrow in self.G_P_A_aggregated_sorted:
            pt = np.array(pt, dtype=float)
            arrow = np.array(arrow, dtype=float)
            norm = np.linalg.norm(arrow)
            if norm > 1e-8:
                dir_vec = arrow / norm
            else:
                dir_vec = arrow  # 零向量时不移动

            # 后移：pt - distance * dir_vec
            back_pt = pt - distance * dir_vec
            # 前移：pt + distance * dir_vec
            forth_pt = pt + distance * dir_vec

            shifted_points.append(tuple(back_pt))
            shifted_points.append(tuple(forth_pt))

        return shifted_points

    # def return_planned_path(self, current_pos):

    #     start = [1, 4, 1] # 起点
    #     gate_points = self.convert_to_planning()

    #     # 创建路径
    #     path_points = []

    #     # 第一圈
    #     path_points.append(current_pos)
    #     path_points.extend(gate_points)
    #     path_points.append(start)
        
    #     # 第二圈
    #     path_points.extend(gate_points)
    #     path_points.append(start)

    #     # 第三圈
    #     path_points.extend(gate_points)
    #     path_points.append(start)

    #     # 第四圈
    #     path_points.extend(gate_points)
    #     path_points.append(start)

    #     return path_points
    
