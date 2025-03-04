import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# KDTree 클래스 정의
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
            return [None, None, points[0]] if points else None

        def get_knn(node, point, k, return_dist_sq, best=None, i=0):
            if best is None:
                best = []

            if node:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]

                if len(best) < k:
                    best.append((dist_sq, node[2]))
                    best.sort(reverse=True)
                elif dist_sq < best[0][0]:
                    best[0] = (dist_sq, node[2])
                    best.sort(reverse=True)

                i = (i + 1) % dim
                first, second = (0, 1) if dx < 0 else (1, 0)
                get_knn(node[first], point, k, return_dist_sq, best, i)
                if len(best) < k or dx ** 2 < best[0][0]:
                    get_knn(node[second], point, k, return_dist_sq, best, i)

            return [(d, p) if return_dist_sq else p for d, p in sorted(best)]

        self._root = make(points)
        self._get_knn = get_knn

    def get_nearest(self, point, return_dist_sq=True):
        nearest = self._get_knn(self._root, point, 1, return_dist_sq)
        return nearest[0] if nearest else None



def load_env_to_vectors(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    ray_hit_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[1], df[2], df[3])]
    return ray_hit_vertices

def load_screen_to_vectors(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    ray_hit_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[3], df[4], df[5])]
    return ray_hit_vertices


env_file_path = os.path.expanduser("~/Downloads/TestEnv.csv")
env = load_env_to_vectors(env_file_path)

screen_file_path = os.path.expanduser("~/Downloads/ScreenVertices.csv")
screen = load_screen_to_vectors(screen_file_path)

# Sample size (e.g.: 10% of all data or maximum 1000)
sample_vertex_size = min(len(screen) // 10, 1000)



# Random sampling
query_points = random.sample(screen, sample_vertex_size)


# KDTree 구축 및 최근접 이웃 검색 시간 측정
kd_tree = KDTree(env, dim=3)
nearest_set_kdtree = []
start_kdtree = time.time()
for idx, query_point in enumerate(query_points):
    nearest_kdtree = kd_tree.get_nearest(query_point)
    nearest_set_kdtree.append(nearest_kdtree)
    if idx % 10 == 0:
        progress = (idx + 1) / len(query_points) * 100
        print(f"KDTree 진행률: {progress:.2f}% 완료")
time_kdtree = time.time() - start_kdtree
print("KDTree 검색 완료")

# 선형 검색을 통한 최근접 이웃 검색 시간 측정
nearest_set_linear = []
start_linear = time.time()
for idx, query_point in enumerate(query_points):
    nearest_linear = min(((sum((x - y) ** 2 for x, y in zip(point, query_point)), point) for point in env))
    nearest_set_linear.append(nearest_linear)
    if idx % 10 == 0:
        progress = (idx + 1) / len(query_points) * 100
        print(f"선형 검색 진행률: {progress:.2f}% 완료")
time_linear = time.time() - start_linear
print("선형 검색 완료")



# 결과 출력
print("KDTree 최근접 이웃:", nearest_kdtree)
print("KDTree 실행 시간:", time_kdtree, "초")
print("선형 검색 최근접 이웃:", nearest_linear)
print("선형 검색 실행 시간:", time_linear, "초")




# Visualize Environment
sample_env_size = min(len(env) // 10, 1000)
sampled_env = random.sample(env, sample_env_size)

# Data preparation
x = [v[0] for v in sampled_env]
y = [v[1] for v in sampled_env]
z = [-v[2] for v in sampled_env]

# 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Drawing dots
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')

# Set axis label
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Change graph title
plt.title(f'3D Visualization of Sampled Data (n={sample_env_size})')

# Add color bar
plt.colorbar(scatter, label='Z value')

# Show graph
plt.show()
