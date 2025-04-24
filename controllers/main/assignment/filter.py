import numpy as np
from collections import defaultdict

class AggregatedExtractor:
    def __init__(self, data_list, gate_point=(4,4,4), dist_thresh=0.5,
                 angle_range=(45,135), cluster_dist=0.8):
        # data_list: list of np.array shape (5,[x,y(,z)]) rows 0=P0, 3=P3, 4=P4
        self.data_list = data_list
        self.points4 = np.array([d[4] for d in data_list])
        self.p0_xy = np.array([d[0,:2] for d in data_list])
        self.p3_xy = np.array([d[3,:2] for d in data_list])
        self.num = len(data_list)
        self.gate_center = np.array(gate_point, float)
        self.T, self.min_ang, self.max_ang = dist_thresh, *angle_range
        self.cluster_dist = cluster_dist
        self.generate_sectors()

    def compute_mask(self):
        d = np.linalg.norm(np.diff(self.points4, axis=0), axis=1)
        return np.concatenate(([True], d < self.T))

    def compute_conditional_idxs(self):
        mask = getattr(self, 'mask', self.compute_mask())
        cond = []
        for i in range(self.num):
            if not mask[i]: continue
            v1 = self.points4[i] - self.gate_center
            v = self.p0_xy[i] - self.p3_xy[i]
            th = np.arctan2(v[1], v[0]) - np.pi/2
            v2 = np.r_[np.cos(th), np.sin(th), 0]
            cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            ang = np.degrees(np.arccos(np.clip(cosang,-1,1)))
            if self.min_ang <= ang <= self.max_ang:
                cond.append(i)
        return cond

    def cluster_idxs(self):
        idxs = self.compute_conditional_idxs()
        clusters, curr = [], []
        for i in idxs:
            if not curr or np.linalg.norm(self.points4[i]-self.points4[curr[-1]])<self.cluster_dist:
                curr.append(i)
            else:
                clusters.append(curr); curr=[i]
        return clusters + ([curr] if curr else [])

    def get_aggregated(self):
        clusters = self.cluster_idxs()
        agg = []
        for cl in clusters:
            pts = self.points4[cl]
            center = pts.mean(0)
            arrows = []
            for i in cl:
                v = self.p0_xy[i] - self.p3_xy[i]
                th = np.arctan2(v[1], v[0]) - np.pi/2
                arrows.append([np.cos(th), np.sin(th), 0])
            agg.append({'Point': center, 'Arrow': np.mean(arrows,0)})
        return agg

    def generate_sectors(self):
        g = np.deg2rad(35)
        n = np.deg2rad(25)
        b = -np.pi - g/2 + g + n
        self.sectors = [(b+i*(g+n), b+i*(g+n)+g) for i in range(5)]

    def _sector(self, xy):
        ang = np.arctan2(xy[1]-self.gate_center[1], xy[0]-self.gate_center[0])
        for i,(s,e) in enumerate(self.sectors):
            if s <= ang <= e:
                return i
        return None

    def sort_aggregated(self):
        agg = self.get_aggregated()
        groups = defaultdict(list)
        for item in agg:
            idx = self._sector(item['Point'][:2])
            if idx is not None:
                groups[idx].append(item['Point'])
        self.gate_positions = {}
        result = []
        for i in sorted(groups):
            avg = np.mean(groups[i],0)
            label = f'Gate{i}'
            self.gate_positions[label] = avg
            result.append((label, avg))
        return result

    def convert_to_planning(self):
        return [tuple(map(lambda x: round(float(x),3),pt)) for _,pt in self.sort_aggregated()]