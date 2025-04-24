# import numpy as np
# import pandas as pd

# class AggregatedExtractor:
#     def __init__(self,
#                  csv_path,
#                  gate_point=(4.0, 4.0, 4.0),
#                  dist_thresh=0.5,
#                  angle_range=(45, 135),
#                  cluster_dist=0.8):
#         self.csv_path       = csv_path
#         self.gate_center    = np.array(gate_point, dtype=float)
#         self.T              = float(dist_thresh)
#         self.min_ang, self.max_ang = angle_range
#         self.cluster_dist   = float(cluster_dist)

#         self._load_and_split()
#         self.filtered        = []
#         self.filterd_for_planning = []
#         self.generate_sector_angle_list()

#     def _load_and_split(self):
#         df = pd.read_csv(self.csv_path)
#         self.df_p0 = df[df['point_index'] == 0].set_index('frame').sort_index()
#         self.df_p3 = df[df['point_index'] == 3].set_index('frame').sort_index()
#         df_p4      = df[df['point_index'] == 4].set_index('frame').sort_index()
#         self.frames = df_p4.index.values
#         coords       = ['x', 'y'] + (['z'] if 'z' in df.columns else [])
#         self.points4 = df_p4[coords].values
#         self.num     = len(self.frames)

#     def compute_mask(self):
#         mask = np.zeros(self.num, dtype=bool)
#         mask[0] = True
#         for i in range(1, self.num):
#             if np.linalg.norm(self.points4[i] - self.points4[i-1]) < self.T:
#                 mask[i] = True
#         self.mask = mask
#         return mask

#     def compute_conditional_idxs(self):
#         if not hasattr(self, 'mask'):
#             self.compute_mask()
#         cond = []
#         for i in range(self.num):
#             if not self.mask[i]:
#                 continue
#             v1 = self.points4[i] - self.gate_center
#             fr = self.frames[i]
#             x0, y0 = self.df_p0.loc[fr, ['x','y']]
#             x3, y3 = self.df_p3.loc[fr, ['x','y']]
#             theta = np.arctan2(y0-y3, x0-x3) - np.pi/2
#             v2 = np.array([np.cos(theta), np.sin(theta), 0.])
#             cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
#             ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
#             if self.min_ang <= ang <= self.max_ang:
#                 cond.append(i)
#         self.cond_idxs = cond
#         return cond

#     def cluster_idxs(self):
#         if not hasattr(self, 'cond_idxs'):
#             self.compute_conditional_idxs()
#         clusters = []
#         if not self.cond_idxs:
#             self.clusters = clusters
#             return clusters
#         current = [self.cond_idxs[0]]
#         for prev, curr in zip(self.cond_idxs, self.cond_idxs[1:]):
#             d = np.linalg.norm(self.points4[curr] - self.points4[prev])
#             if d < self.cluster_dist:
#                 current.append(curr)
#             else:
#                 clusters.append(current)
#                 current = [curr]
#         clusters.append(current)
#         self.clusters = clusters
#         return clusters

#     def get_aggregated(self):
#         if not hasattr(self, 'clusters'):
#             self.cluster_idxs()
#         result = []
#         for clust in self.clusters:
#             pts = self.points4[clust]
#             center = pts.mean(axis=0)
#             vecs = []
#             for i in clust:
#                 fr = self.frames[i]
#                 x0, y0 = self.df_p0.loc[fr, ['x','y']]
#                 x3, y3 = self.df_p3.loc[fr, ['x','y']]
#                 th = np.arctan2(y0-y3, x0-x3) - np.pi/2
#                 vecs.append([np.cos(th), np.sin(th), 0.])
#             avg_vec = np.array(vecs).mean(axis=0)
#             result.append({"Point": center, "Arrow": avg_vec})
#         self.filtered = result
#         return result

#     def is_in_range(self, angle, range):
#         return range[0] <= angle <= range[1]

#     def generate_sector_angle_list(self):
#         angle_gate = np.deg2rad(35)
#         angle_none = np.deg2rad(60) - angle_gate
#         angle_start = -np.pi - angle_gate/2
#         ang = angle_start + angle_gate + angle_none
#         gate_angle_list = []
#         for i in range(5):
#             gate_angle_list.append([ang, ang + angle_gate])
#             ang += angle_gate + angle_none
#         self.gate_angle_list = gate_angle_list

#     def _compute_sector(self, point_xy):
#         v   = np.array(point_xy) - self.gate_center[:2]
#         ang = np.arctan2(v[1], v[0])
#         sector = None
#         for i, ang_range in enumerate(self.gate_angle_list):
#             if self.is_in_range(ang, ang_range):
#                 sector = i
#                 break
#         label = f"Gate{sector}" if sector is not None else "None"
#         return sector, label

#     def sort_aggregated(self):
#         if not hasattr(self, 'filtered') or not self.filtered:
#             self.get_aggregated()
#         annotated = []
#         for item in self.filtered:
#             pt_xy = item["Point"][:2]
#             sec, lbl = self._compute_sector(pt_xy)
#             annotated.append((sec, lbl, item))
#         filtered = [t for t in annotated if t[1].startswith("Gate")]
#         filtered.sort(key=lambda x: x[0])
#         # 为每个扇区计算平均坐标
#         sector_points = {}
#         for _, lbl, data in filtered:
#             sector_points.setdefault(lbl, []).append(data["Point"])
#         self.filtered = []
#         self.gate_positions = {}
#         # 按 Gate0~Gate4 顺序输出
#         for lbl in sorted(sector_points.keys(), key=lambda x: int(x.replace("Gate", ""))):
#             pts = np.array(sector_points[lbl])
#             avg_pt = pts.mean(axis=0)
#             self.gate_positions[lbl] = avg_pt
#             self.filtered.append((lbl, avg_pt))
#         return self.filtered

#     def convert_to_planning(self):
#         if self.filtered is not None:
#             for item in self.filtered:
#                 point = item[1]
#                 point = tuple(float(coord) for coord in point)
#                 point = tuple(round(coord, 3) for coord in point)
#                 self.filterd_for_planning.append(point)
#         return self.filterd_for_planning

# # ===== 使用示例 =====
# if __name__ == "__main__":
#     extractor = AggregatedExtractor('target_positions.csv', gate_point=(4,4,4))
#     extractor.get_aggregated()
#     gate_positions = extractor.sort_aggregated()
#     for label, pos in gate_positions:
#         print(f"{label} gate position: {pos}")

import numpy as np
import pandas as pd
from collections import defaultdict

class AggregatedExtractor:
    def __init__(self, csv_path, gate_point=(4,4,4), dist_thresh=0.5,
                 angle_range=(45,135), cluster_dist=0.8):
        self.gate_center = np.array(gate_point, float)
        self.T, self.min_ang, self.max_ang = dist_thresh, *angle_range
        self.cluster_dist = cluster_dist
        self._load_data(csv_path)
        self.generate_sector_angles()

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

    def compute_conditional_idxs(self):
        mask = getattr(self, 'mask', self.compute_mask())
        cond = []
        for i, fr in enumerate(self.frames):
            if not mask[i]: continue
            v1 = self.points4[i] - self.gate_center
            xy0, xy3 = self.df_p0.loc[fr], self.df_p3.loc[fr]
            theta = np.arctan2(*(xy0.values - xy3.values)[::-1]) - np.pi/2
            v2 = np.r_[np.cos(theta), np.sin(theta), 0]
            ang = np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8), -1,1)))
            if self.min_ang <= ang <= self.max_ang:
                cond.append(i)
        return cond

    def cluster_idxs(self, idxs=None):
        idxs = idxs or self.compute_conditional_idxs()
        clusters, curr = [], [idxs[0]] if idxs else []
        for a,b in zip(idxs, idxs[1:]):
            if np.linalg.norm(self.points4[b]-self.points4[a])<self.cluster_dist:
                curr.append(b)
            else:
                clusters.append(curr); curr=[b]
        return clusters + [curr] if curr else []

    def get_aggregated(self):
        clusters = self.cluster_idxs()
        agg = []
        for cl in clusters:
            pts = self.points4[cl]
            center = pts.mean(0)
            arrows = []
            for i in cl:
                fr = self.frames[i]
                v = self.df_p0.loc[fr]-self.df_p3.loc[fr]
                th = np.arctan2(v.y, v.x)-np.pi/2
                arrows.append([np.cos(th),np.sin(th),0])
            agg.append({'Point':center,'Arrow':np.mean(arrows,0)})
        return agg

    def generate_sector_angles(self):
        gate, none = np.deg2rad(35), np.deg2rad(25)
        base = -np.pi - gate/2 + gate + none
        self.sectors = [(base+i*(gate+none), base+i*(gate+none)+gate) for i in range(5)]

    def _sector(self, pt):
        ang = np.arctan2(*(pt-self.gate_center[:2])[::-1])
        for i,(s,e) in enumerate(self.sectors):
            if s<=ang<=e:
                return f'Gate{i}', i
        return None, None

    def sort_aggregated(self):
        agg = self.get_aggregated()
        groups = defaultdict(list)
        for item in agg:
            lbl, idx = self._sector(item['Point'][:2])
            if lbl: groups[idx].append(item['Point'])
        sorted_keys = sorted(groups)
        self.gate_positions = {f'Gate{i}':np.mean(groups[i],0) for i in sorted_keys}
        return list(self.gate_positions.items())

    def convert_to_planning(self):
        return [tuple(map(lambda x:round(float(x),3),pt)) for _,pt in self.sort_aggregated()]