import numpy as np
import pandas as pd

class AggregatedExtractor:
    def __init__(self,
                 csv_path,
                 gate_point=(4.0, 4.0),
                 dist_thresh=0.5,
                 angle_range=(45, 135),
                 cluster_dist=0.8):
        """
        Parameters
        ----------
        csv_path : str
            包含 point_index、frame、x、y 的 CSV 文件路径
        gate_point : tuple of float
            计算 v1 时的起点 (默认 (4,4))
        dist_thresh : float
            生成 mask 时，相邻帧 P4 距离阈值
        angle_range : tuple of float
            角度筛选范围，单位度 (min_angle, max_angle)
        cluster_dist : float
            聚类时，相邻点归为一簇的最大距离阈值
        """
        self.csv_path = csv_path
        self.gate = np.array(gate_point, dtype=float)
        self.T = float(dist_thresh)
        self.min_ang, self.max_ang = angle_range
        self.cluster_dist = float(cluster_dist)
        
        # 读取并分组
        self._load_and_split()

    def _load_and_split(self):
        df = pd.read_csv(self.csv_path)
        self.df_p0 = df[df['point_index'] == 0].set_index('frame').sort_index()
        self.df_p3 = df[df['point_index'] == 3].set_index('frame').sort_index()
        df_p4 = df[df['point_index'] == 4].set_index('frame').sort_index()
        self.frames = df_p4.index.values
        self.points4 = df_p4[['x','y']].values
        self.num = len(self.frames)

    def compute_mask(self):
        mask = np.zeros(self.num, dtype=bool)
        mask[0] = True
        for i in range(1, self.num):
            if np.hypot(*(self.points4[i] - self.points4[i-1])) < self.T:
                mask[i] = True
        self.mask = mask
        return mask

    def compute_conditional_idxs(self):
        if not hasattr(self, 'mask'):
            self.compute_mask()

        cond = []
        for i in range(self.num):
            if not self.mask[i]:
                continue
            # v1: 从 gate 指向 points4[i]
            v1 = self.points4[i] - self.gate
            # v2: 法向量
            fr = self.frames[i]
            x0, y0 = self.df_p0.loc[fr, ['x','y']]
            x3, y3 = self.df_p3.loc[fr, ['x','y']]
            theta = np.arctan2(y0-y3, x0-x3) - np.pi/2
            v2 = np.array([np.cos(theta), np.sin(theta)])
            # 夹角
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
            ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            # 叉积方向 (z 分量)
            v1_3 = np.array([v1[0], v1[1], 0.])
            v2_3 = np.array([v2[0], v2[1], 0.])
            cross_z = np.cross(v1_3, v2_3)[2]
            # 同时满足角度范围 且 叉积 > 0
            if self.min_ang <= ang <= self.max_ang and cross_z > 0:
                cond.append(i)

        self.cond_idxs = cond
        return cond

    def cluster_idxs(self):
        if not hasattr(self, 'cond_idxs'):
            self.compute_conditional_idxs()

        clusters = []
        if not self.cond_idxs:
            self.clusters = clusters
            return clusters

        current = [self.cond_idxs[0]]
        for prev, curr in zip(self.cond_idxs, self.cond_idxs[1:]):
            d = np.hypot(*(self.points4[curr] - self.points4[prev]))
            if d < self.cluster_dist:
                current.append(curr)
            else:
                clusters.append(current)
                current = [curr]
        clusters.append(current)
        self.clusters = clusters
        return clusters

    def get_aggregated(self):
        """
        返回一个列表，每个元素是字典：
          {
            "Point": np.array([center_x, center_y]),
            "arrow": np.array([avg_dx, avg_dy])
          }
        """
        if not hasattr(self, 'clusters'):
            self.cluster_idxs()

        result = []
        for clust in self.clusters:
            pts = self.points4[clust]
            xm, ym = pts.mean(axis=0)
            # 计算平均箭头方向
            vecs = []
            for i in clust:
                fr = self.frames[i]
                x0, y0 = self.df_p0.loc[fr, ['x','y']]
                x3, y3 = self.df_p3.loc[fr, ['x','y']]
                th = np.arctan2(y0-y3, x0-x3) - np.pi/2
                vecs.append([np.cos(th), np.sin(th)])
            avg_dx, avg_dy = np.array(vecs).mean(axis=0)
            result.append({
                "Point": np.array([xm, ym]),
                "Arrow": np.array([avg_dx, avg_dy])
            })
        return result


# ===== 使用示例 =====
if __name__ == '__main__':
    extractor = AggregatedExtractor('target_positions.csv')
    aggregated_list = extractor.get_aggregated()
    for item in aggregated_list:
        print("Point:", item["Point"], " arrow:", item["arrow"])
