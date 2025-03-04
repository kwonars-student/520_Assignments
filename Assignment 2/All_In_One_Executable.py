import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

def load_env_to_vectors(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    ray_hit_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[1], df[2], df[3])]
    return ray_hit_vertices

def load_screen_to_vectors(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    indexes = [[float(i), float(j)] for i, j in zip(df[1], df[2])]
    count_screens = sum(1 for index in indexes if index == [0, 0])
    screen_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[3], df[4], df[5])]
    if count_screens > 0:
        split_screen_vertices = np.array_split(screen_vertices, count_screens)
    else:
        split_screen_vertices = []

    return split_screen_vertices

env_file_path = os.path.expanduser(r'C:\Users\kwonars\Desktop\COSC520\Assignments, PT, Project\Assignment 2\TestEnv.csv')
env = load_env_to_vectors(env_file_path)

screen_file_path = os.path.expanduser(r'C:\Users\kwonars\Desktop\COSC520\Assignments, PT, Project\Assignment 2\ScreenVertices.csv')
all_screens = load_screen_to_vectors(screen_file_path)

# Sample size (e.g.: 10% of all data or maximum 1000)
sample_screen_number = min(len(all_screens) // 10, 3)



# Random sampling
query_screens = random.sample(all_screens, sample_screen_number)


# 유클리드 거리 계산 함수 (Linear Search 및 KD-Tree 모두 사용)
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))


# Linear Search (전체 탐색)
def linear_search_nearest_neighbor(points, target):
    """
    선형 탐색으로 가장 가까운 점 찾기
    """
    nearest = None
    min_distance = float('inf')  # 무한대로 초기화

    for point in points:
        distance = euclidean_distance(target, point)
        if distance < min_distance:
            min_distance = distance
            nearest = point

    return nearest


# KD-Tree 노드 클래스
class Node:
    def __init__(self, point=None, left=None, right=None, axis=None):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis


# KD-Tree 생성 함수
def build_kd_tree(points, depth=0):
    n = len(points)
    if n == 0:
        return None

    axis = depth % 3
    sorted_points = sorted(points, key=lambda point: point[axis])
    median_index = n // 2
    median_point = sorted_points[median_index]

    node = Node(point=median_point, axis=axis)

    left_points = sorted_points[:median_index]
    right_points = sorted_points[median_index+1:]

    node.left = build_kd_tree(left_points, depth + 1)
    node.right = build_kd_tree(right_points, depth + 1)

    return node


# KD-Tree 최근접 이웃 검색 함수
def nearest_neighbor_search(root, target_point, current_best=None, depth=0):
    if root is None:
        return current_best

    axis = depth % 3
    current_point = root.point
    distance = euclidean_distance(target_point, current_point)

    if current_best is None or distance < euclidean_distance(target_point, current_best.point):
        current_best = root

    if target_point[axis] < current_point[axis]:
        next_branch = root.left
        opposite_branch = root.right
    else:
        next_branch = root.right
        opposite_branch = root.left

    current_best = nearest_neighbor_search(next_branch, target_point, current_best, depth + 1)

    # 평면 거리 계산 시 axis에 따라 조건 변경
    if euclidean_distance( (target_point[0], target_point[1], current_point[2]) if axis==2 else
                            ((target_point[0], current_point[1], target_point[2]) if axis==1 else
                             (current_point[0], target_point[1], target_point[2])), target_point) < euclidean_distance(target_point, current_best.point):
    #if abs(target_point[axis] - current_point[axis]) < euclidean_distance(target_point, current_best.point):
        current_best = nearest_neighbor_search(opposite_branch, target_point, current_best, depth + 1)

    return current_best


# 데이터 생성
def generate_random_points(num_points):
    return [(random.random(), random.random(), random.random()) for _ in range(num_points)]


# # 실험 설정
# num_points = 10000 # 점의 개수
# num_queries = 100   # 검색 쿼리 개수
#
# # 1. 데이터 생성
# points = generate_random_points(num_points)

# 2. KD-Tree 생성
start_time = time.time()
kd_tree_root = build_kd_tree(env)
kd_tree_build_time = time.time() - start_time
print(f"KD-Tree build time: {kd_tree_build_time:.4f} seconds")

# # 3. 검색 쿼리 생성
# queries = generate_random_points(num_queries)

# 4. Linear Search 시간 측정
start_time = time.time()
min_distance_linear = []
avg_distance_linear = []
for screen in query_screens:
    for query in screen:
        closest_point = linear_search_nearest_neighbor(env, query)
        min_distance_linear.append(euclidean_distance(query, closest_point))
    avg_distance_linear.append(np.mean(min_distance_linear))
    min_distance_linear = []
linear_search_time = time.time() - start_time
print(f"Linear search time (for {len(query_screens)*len(screen)} queries): {linear_search_time:.4f} seconds")

# 5. KD-Tree Search 시간 측정
start_time = time.time()
min_distance_kd = []
avg_distance_kd = []
for screen in query_screens:
    for query in screen:
        closest_point = nearest_neighbor_search(kd_tree_root, query).point
        min_distance_kd.append(euclidean_distance(query, closest_point))
    avg_distance_kd.append(np.mean(min_distance_kd))
    min_distance_kd = []
kd_tree_search_time = time.time() - start_time
print(f"KD-Tree search time (for {len(query_screens)*len(screen)} queries): {kd_tree_search_time:.4f} seconds")

# 결과 비교
print(f"\nSpeedup (Linear / KD-Tree): {linear_search_time / kd_tree_search_time:.2f}x")

print(avg_distance_linear)
print(avg_distance_kd)


# # Visualize Environment
# sample_env_size = min(len(env) // 10, 1000)
# sampled_env = random.sample(env, sample_env_size)
#
# # Data preparation
# x = [v[0] for v in sampled_env]
# y = [v[1] for v in sampled_env]
# z = [-v[2] for v in sampled_env]
#
# # 3D figure
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Drawing dots
# scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
#
# # Set axis label
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Change graph title
# plt.title(f'3D Visualization of Sampled Data (n={sample_env_size})')
#
# # Add color bar
# plt.colorbar(scatter, label='Z value')
#
# # Show graph
# plt.show()
