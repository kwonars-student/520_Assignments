import pandas as pd
import numpy as np
import os
import time


class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) > 1:
            self._swap(0, len(self.heap) - 1)
            item = self.heap.pop()
            self._sift_down(0)
        elif self.heap:
            item = self.heap.pop()
        else:
            item = None
        return item

    def top(self):
        return self.heap[0] if self.heap else None

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        if idx > 0 and self.heap[idx][0] < self.heap[parent][0]:
            self._swap(idx, parent)
            self._sift_up(parent)

    def _sift_down(self, idx):
        child = 2 * idx + 1
        if child < len(self.heap):
            if child + 1 < len(self.heap) and self.heap[child + 1][0] < self.heap[child][0]:
                child += 1
            if self.heap[child][0] < self.heap[idx][0]:
                self._swap(child, idx)
                self._sift_down(child)

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __len__(self):
        return len(self.heap)

# https://github.com/Vectorized/Python-KD-Tree/blob/master/kd_tree.py
class KDTree:
    def __init__(self, points, dim, dist_sq_func=None):
        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 for i, x in enumerate(a))

        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), points[m]]
            if len(points) == 1:
                return [None, None, points[0]]

        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heap.push((-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap.top()[0]:
                    heap.pop()
                    heap.push((-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap.top()[0])]:
                    get_knn(node[b], point, k, return_dist_sq, heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] for h in sorted(heap.heap)][::-1]

        self._get_knn = get_knn
        self._root = make(points)

    def get_nearest(self, point, return_dist_sq=True):
        heap = MinHeap()
        l = self._get_knn(self._root, point, 1, return_dist_sq, heap)
        return l[0] if len(l) else None


def load_csv_to_vectors(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    ray_hit_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[0], df[1], df[2])]
    return ray_hit_vertices


file_path = os.path.expanduser("~/Downloads/TestEnv.csv")
ray_vertices = load_csv_to_vectors(file_path)

# KDTree search timing
tree = KDTree(ray_vertices, dim=3)
query_point = [1.0, 2.0, 3.0]
start_time = time.time()
nearest = tree.get_nearest(query_point)
kdtree_time = time.time() - start_time

distance = np.sqrt(nearest[0]) if nearest else None
nearest_point = nearest[1] if nearest else None
print(f"KDTree - Nearest Point: {nearest_point}, Distance: {distance}, Time: {kdtree_time:.6f} sec")

# Linear search timing
start_time = time.time()
min_dist = float('inf')
nearest_point_linear = None
for point in ray_vertices:
    dist = sum((query_point[i] - point[i]) ** 2 for i in range(3))
    if dist < min_dist:
        min_dist = dist
        nearest_point_linear = point
linear_time = time.time() - start_time

distance_linear = np.sqrt(min_dist)
print(
    f"Linear Search - Nearest Point: {nearest_point_linear}, Distance: {distance_linear}, Time: {linear_time:.6f} sec")
