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
        self.P_A_aggregated                = None  # 聚合后点 + 方向        dict 数据
        self.G_P_A_aggregated_sorted       = None  # 排序后的聚合结果
        self.G_P_A_aggregated_sorted_comp  = None  # 补偿后的聚合结果
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

    # def sort_aggregated(self):
    #     groups = defaultdict(list)
    #     for item in self.P_A_aggregated:
    #         label, idx = self._sector(item['Point'][:2])
    #         if idx is not None:
    #             groups[idx].append((item['Point'], item['Arrow']))

    #     result = []
    #     for idx in sorted(groups):
    #         pts, arrs = zip(*groups[idx])
    #         mean_pt = np.mean(pts, axis=0)
    #         mean_arr = np.mean(arrs, axis=0)
    #         result.append((f'Gate{idx}', mean_pt, mean_arr))

    #     self.G_P_A_aggregated_sorted = result



    # def convert_to_planning(self):
    #     # 返回 [(x,y,z), ...]
    #     return [tuple(np.round(pt, 3)) for _, pt, _ in self.G_P_A_aggregated_sorted]

    # def convert_to_planning_shift(self, shift = 0.3):
    #     self.G_P_A_aggregated_sorted_shift = []
    #     for label, center, arrow in self.G_P_A_aggregated_sorted:
    #         angle = np.arctan2(arrow[1], arrow[0])
    #         new_dir = angle - np.pi/2
    #         new_vec = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
    #         new_pt = center + shift * new_vec
    #         point = tuple(np.round(new_pt, 3))
    #         self.G_P_A_aggregated_sorted_shift.append((label, point, arrow))
    #     return [pt for _, pt, _ in self.G_P_A_aggregated_sorted_shift]
    

    # 基于字典型数据
    def sort_aggregated(self):
        groups = defaultdict(list)
        for item in self.P_A_aggregated:
            label, idx = self._sector(item['Point'][:2])
            if idx is not None:
                groups[idx].append((item['Point'], item['Arrow']))

        # 建立一个按照 idx 排序的 dict：GateX → (mean_pt, mean_arr)
        d = {}
        for idx in sorted(groups):
            pts, arrs = zip(*groups[idx])
            mean_pt  = np.mean(pts,  axis=0)
            mean_arr = np.mean(arrs, axis=0)
            d[f'Gate{idx}'] = (mean_pt, mean_arr)

        # 存成字典，保留 insertion order
        self.G_P_A_aggregated_sorted = d

    def convert_to_planning(self):
        """
        返回 [(x,y,z), ...]，顺序与原先按 Gate1, Gate2, ... 排序保持一致
        """
        # 如果 Gate 标签是 Gate<number>，下面这句会按数字排序：
        items = sorted(
            self.G_P_A_aggregated_sorted.items(),
            key=lambda kv: int(kv[0].replace('Gate', ''))
        )
        return [
            tuple(np.round(center, 3))
            for _, (center, _) in items
        ]



    def convert_to_planning_with_compensate(self, compensate_dict):

        # 原始数据
        self.G_P_A_aggregated_sorted

        # 计算补偿后的坐标
        self.G_P_A_aggregated_sorted_comp = {}
        for label, (center, arrow) in self.G_P_A_aggregated_sorted.items():
            if label in compensate_dict:
                center = np.array(center) + np.array(compensate_dict[label])
            self.G_P_A_aggregated_sorted_comp[label] = (center, arrow)
        
        # 返回 list 数据
        items = sorted(
            self.G_P_A_aggregated_sorted_comp.items(),
            key=lambda kv: int(kv[0].replace('Gate', ''))
        )

        return [
            tuple(np.round(center, 3))
            for _, (center, _) in items
        ]


