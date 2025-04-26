import numpy as np
import pandas as pd
from collections import defaultdict

class AggregatedExtractor:
    def __init__(self, csv_path, gate_point=(4,4,4), dist_thresh=0.5,
                 angle_range=(45,135), cluster_dist=0.8):
        self.gate_center = np.array(gate_point, float)
        self.T, self.min_ang, self.max_ang = dist_thresh, *angle_range
        self.cluster_dist = cluster_dist

        # 自建数据
        self.data_filtered              = None  # 点 + 角度
        self.data_filtered_sorted       = None  # 排序后的点 + 角度 
        self.data_filtered_sorted_shift = None

        # 读取数据
        self._load_data(csv_path)        # 加载数据
        self.generate_sector_angles()    # 生成分区表
        self.compute_conditional_idxs()  # 计算并聚类
        self.sort_aggregated()           # 排序聚类点

    def _load_data(self, path):
        df = pd.read_csv(path)
        self.df_p0 = df.query('point_index==0').set_index('frame')[['x','y']]
        self.df_p3 = df.query('point_index==3').set_index('frame')[['x','y']]
        p4 = df.query('point_index==4').sort_values('frame')
        self.frames = p4['frame'].values
        coords = ['x','y'] + (['z'] if 'z' in p4.columns else [])
        self.points4 = p4[coords].values

    def compute_mask(self):
        dists = np.linalg.norm(np.diff(self.points4, axis=0), axis=1)
        mask = np.concatenate(([True], dists < self.T))
        return mask

    # 判断角度 + 叉乘，计算索引并聚合点与箭头方向
    def compute_conditional_idxs(self):
        mask = getattr(self, 'mask', self.compute_mask())
        idxs = []
        arrow_dict = {}
        # 1) 筛选满足距离和角度条件的索引并计算箭头方向
        for i, fr in enumerate(self.frames):
            if not mask[i]: continue
            v1 = self.points4[i] - self.gate_center
            xy0, xy3 = self.df_p0.loc[fr], self.df_p3.loc[fr]
            theta = np.arctan2(*(xy0.values - xy3.values)[::-1]) - np.pi/2
            v2 = np.r_[np.cos(theta), np.sin(theta), 0]
            # 检查夹角
            ang = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                        -1, 1
                    )
                )
            )
            if self.min_ang <= ang <= self.max_ang:
                idxs.append(i)
                # 如果 v1 和 v2 的叉乘指向下方（z<0），则反转箭头方向
                cross_z = np.cross(v1, v2)[2]
                arrow = -v2 if cross_z < 0 else v2
                arrow_dict[i] = arrow

        # 2) 基于空间距离聚类
        clusters, curr = [], [idxs[0]] if idxs else []
        for a, b in zip(idxs, idxs[1:]):
            if np.linalg.norm(self.points4[b] - self.points4[a]) < self.cluster_dist:
                curr.append(b)
            else:
                clusters.append(curr)
                curr = [b]
        if curr:
            clusters.append(curr)

        # 3) 聚合每个聚类的中心点和平均箭头
        agg = []
        for cl in clusters:
            pts = self.points4[cl]
            center = pts.mean(0)
            arrows = np.array([arrow_dict[i] for i in cl])
            avg_arrow = arrows.mean(0)
            agg.append({'Point': center, 'Arrow': avg_arrow})

        self.data_filtered = agg
        return agg

    ############################################## sector 排序 #######################################
    def generate_sector_angles(self):
        gate, none = np.deg2rad(35), np.deg2rad(25)
        base = -np.pi - gate/2 + gate + none
        self.sectors = [(base + i*(gate + none), base + i*(gate + none) + gate) for i in range(5)]

    def _sector(self, pt):
        ang = np.arctan2(*(pt - self.gate_center[:2])[::-1])
        for i, (s, e) in enumerate(self.sectors):
            if s <= ang <= e:
                return f'Gate{i}', i
        return None, None

    # sector 排序
    def sort_aggregated(self):
        points_arrows = self.data_filtered
        groups = defaultdict(list)

        # 聚集 (Point, Arrow) 对
        for item in points_arrows:
            label, idx = self._sector(item['Point'][:2])
            if idx is None:
                continue
            groups[idx].append((item['Point'], item['Arrow']))

        result = []
        for idx in sorted(groups):
            pts, arrs = zip(*groups[idx])
            mean_pt  = np.mean(pts,  axis=0)
            mean_arr = np.mean(arrs,  axis=0)
            result.append((f'Gate{idx}', mean_pt, mean_arr))

        self.data_filtered_sorted = result

    def convert_to_planning(self):
        return [tuple(map(lambda x: round(float(x), 3), pt)) for _, pt, _ in self.data_filtered_sorted]

    def convert_to_planning_shift(self):
        self.data_filtered_sorted_shift = []
        for item in self.data_filtered_sorted:
            # 拆分数据
            label, center, arrow = item
            # 计算新的坐标
            angle   = np.arctan2(arrow[1], arrow[0])
            new_dir = angle - np.pi/2
            new_dir = np.array([np.cos(new_dir), np.sin(new_dir), 0.])
            offset  = 0.5 * new_dir
            new_pt  = center + offset
            # 更新数据
            point   = (
                round(float(new_pt[0]), 3),
                round(float(new_pt[1]), 3),
                round(float(new_pt[2]), 3)
            )
            self.data_filtered_sorted_shift.append((label, point, arrow))

        return [tuple(map(lambda x: round(float(x), 3), pt)) for _, pt, _ in self.data_filtered_sorted_shift]