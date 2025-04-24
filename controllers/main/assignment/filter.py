import numpy as np
import pandas as pd

class AggregatedExtractor:
    def __init__(self,
                 csv_path,
                 gate_point=(4.0, 4.0, 4.0),
                 dist_thresh=0.5,
                 angle_range=(45, 135),
                 cluster_dist=0.8):
        # ... （省略前面已有的 __init__、_load_and_split、compute_mask、compute_conditional_idxs、cluster_idxs）
        self.csv_path       = csv_path
        self.gate_center           = np.array(gate_point, dtype=float)
        self.T              = float(dist_thresh)
        self.min_ang, self.max_ang = angle_range
        self.cluster_dist   = float(cluster_dist)
        self.filterd        = []
        self.filterd_for_planning = []
        self._load_and_split()

        self.generate_sector_angle_list()  # 生成扇区角度列表

    def _load_and_split(self):

        df = pd.read_csv(self.csv_path)

        # 按索引分离 P0, P3, P4
        self.df_p0 = df[df['point_index'] == 0].set_index('frame').sort_index()
        self.df_p3 = df[df['point_index'] == 3].set_index('frame').sort_index()
        df_p4      = df[df['point_index'] == 4].set_index('frame').sort_index()
        self.frames = df_p4.index.values

        # 如果 CSV 包含 z，则读取三列，否则退回到二维
        coords = ['x', 'y'] + (['z'] if 'z' in df.columns else [])
        self.points4 = df_p4[coords].values
        self.num     = len(self.frames)


    def compute_mask(self):
        mask = np.zeros(self.num, dtype=bool)
        mask[0] = True
        # 使用 np.linalg.norm 来支持任意维度的距离计算
        for i in range(1, self.num):
            if np.linalg.norm(self.points4[i] - self.points4[i-1]) < self.T:
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
            # v1: 从 gate 指向 points4[i]（现在可能是三维向量）
            v1 = self.points4[i] - self.gate_center

            # 下面计算仍然在二维平面上，法向量 z 分量设为 0
            fr = self.frames[i]
            x0, y0 = self.df_p0.loc[fr, ['x','y']]
            x3, y3 = self.df_p3.loc[fr, ['x','y']]
            theta = np.arctan2(y0-y3, x0-x3) - np.pi/2
            v2 = np.array([np.cos(theta), np.sin(theta), 0.])

            # 计算夹角
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
            ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            # 叉积方向 (z 分量)
            cross_z = np.cross(v1, v2)[2]

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
            d = np.linalg.norm(self.points4[curr] - self.points4[prev])
            if d < self.cluster_dist:
                current.append(curr)
            else:
                clusters.append(current)
                current = [curr]
        clusters.append(current)
        self.clusters = clusters
        return clusters

    def get_aggregated(self):
        # 完整三维版聚合（同前面那版）
        if not hasattr(self, 'clusters'):
            self.cluster_idxs()

        result = []
        for clust in self.clusters:
            pts = self.points4[clust]
            center = pts.mean(axis=0)
            vecs = []
            for i in clust:
                fr = self.frames[i]
                x0, y0 = self.df_p0.loc[fr, ['x','y']]
                x3, y3 = self.df_p3.loc[fr, ['x','y']]
                th = np.arctan2(y0-y3, x0-x3) - np.pi/2
                vecs.append([np.cos(th), np.sin(th), 0.])
            avg_vec = np.array(vecs).mean(axis=0)
            result.append({"Point": center, "Arrow": avg_vec})
        self.filterd = result
        return result


    ########################################################## 扇区处理部分
    def is_in_range(self, angle, range):
        if range[0] <= angle and angle <= range[1]:
            return True
        else:
            return False

    def generate_sector_angle_list(self):
        
        angle_gate = np.deg2rad(35)
        angle_none = np.deg2rad(60) - angle_gate

        angle_start = -np.pi - angle_gate/2

        ang = angle_start + angle_gate + angle_none

        gate_angle_list = []

        for i in range(5):
            angle_rangle = [ang, ang + angle_gate]
            gate_angle_list.append(angle_rangle)
            ang += angle_gate + angle_none
        
        self.gate_angle_list = gate_angle_list

    def _compute_sector(self, point_xy):

        # 1) 计算向量
        v   = np.array(point_xy) - self.gate_center[:2]
        ang = np.arctan2(v[1], v[0])

        # 2) 计算 sector
        def get_sector(angle):
            i = 0
            for range in self.gate_angle_list:
                if self.is_in_range(angle, range):
                    return i
                i += 1
            return None
        
        sector = get_sector(ang)

        # 3) 标签
        if sector == None:
            label = "None"
        else:
            label = f"Gate{sector}"

        return sector, label

    def sort_aggregated(self):
        """
        对 self.filterd 中的点打扇区标签并按 CCW 顺序排序，
        只保留 Start 和 Gate1~Gate5（剔除 None）。
        返回一个列表：[(label, {"Point":..., "Arrow":...}), ...]
        """
        if not hasattr(self, 'filterd') or not self.filterd:
            self.get_aggregated()

        annotated = []
        for item in self.filterd:
            # 只看 XY 做扇区
            pt_xy = item["Point"][:2]
            sec, lbl = self._compute_sector(pt_xy)
            annotated.append((sec, lbl, item))

        # 只保留 Start 及 Gate1~Gate5
        filtered = [t for t in annotated if t[1] != "None"]
        # 按扇区号排序
        filtered.sort(key=lambda x: x[0])

        # 返回 (label, data) 格式
        self.filterd = [(lbl, data) for sec,lbl,data in filtered]
        return self.filterd
    
    def convert_to_planning(self):
        """
        返回一个三维点列表，直接用于路径规划
        """
        if self.filterd is not None:

            for item in self.filterd:

                point = item["Point"]
                point = tuple(point)                              # 将三维坐标转换为 tuple
                point = tuple(float(coord) for coord in point)    # 取消 numpy float64 格式
                point = tuple(round(coord, 3) for coord in point) # 保留三位小数

                # 将三维坐标转换为 tuple 并添加到列表中
                self.filterd_for_planning.append(point)

        return self.filterd_for_planning


# ===== 使用示例 =====
if __name__ == "__main__":
    extractor = AggregatedExtractor('target_positions.csv',
                                    gate_point=(4,4,4))
    extractor.get_aggregated()
    ordered = extractor.sort_aggregated()

    for label, item in ordered:
        print(f"{label}: Point={item['Point']}, Arrow={item['Arrow']}")
