import math

# Predefined points
points = [
    [1.0, 2.0, 3.0],  # Point A
    [4.0, 5.0, 6.0],  # Point B
    [7.0, 8.0, 9.0],  # Point C
]

# Predefined target points for testing
target_points = [
    [5.0, 5.0, 5.0],  # Target point 1 (expected nearest is B)
    [2.0, 2.0, 2.0],  # Target point 2 (expected nearest is A)
    [8.0, 8.0, 8.0],  # Target point 3 (expected nearest is C)
]

# Extreme test cases
extreme_target_points = [
    [100.0, 100.0, 100.0],  # Very far target (expected nearest is C)
    [0.0, 0.0, 0.0],        # Origin (expected nearest is A)
    [-1.0, -1.0, -1.0],     # Negative coordinates (expected nearest is A)
]

print("\n")

################################################
# 1. Test euclidean_distance function
################################################
print(
  "1. Test euclidean_distance function"
)

def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

for i in range(len(points)):
    for j in range(i + 1, len(points)):
        distance = euclidean_distance(points[i], points[j])  # Calculate distance between points[i] and points[j]
        print(f"    Distance between Point {i+1} and Point {j+1}: {distance:.3f}")  # Actual value
        # Expected:
        # Distance between A and B: sqrt((1.0 - 4.0)^2 + (2.0 - 5.0)^2 + (3.0 - 6.0)^2) = sqrt(27) ≈ 5.196
        # Distance between A and C: sqrt((1.0 - 7.0)^2 + (2.0 - 8.0)^2 + (3.0 - 9.0)^2) = sqrt(108) ≈ 10.392
        # Distance between B and C: sqrt((4.0 - 7.0)^2 + (5.0 - 8.0)^2 + (6.0 - 9.0)^2) = sqrt(27) ≈ 5.196

print("\n")

################################################################################################
# 2. Test linear_search_nearest_neighbor function with multiple target points
################################################################################################
print(
  "2. Test linear_search_nearest_neighbor function with multiple target points"
)

def linear_search_nearest_neighbor(points, target):
    nearest = None
    min_distance = float('inf')  # Initialize minimum distance as infinity

    # Iterate through each point to find the nearest
    for point in points:
        distance = euclidean_distance(target, point)  # Calculate distance to the target point
        if distance < min_distance:  # If found a closer point
            min_distance = distance  # Update minimum distance
            nearest = point  # Update nearest point

    return nearest


for target_point in target_points + extreme_target_points:
    nearest_point_linear = linear_search_nearest_neighbor(points, target_point)  # Find nearest point for the current target
    print(f"    Nearest point (Linear Search) to {target_point}: {nearest_point_linear}")  # Actual value
    # Expected:
    # For [5.0, 5.0, 5.0]: [4.0, 5.0, 6.0] (Point B)
    # For [2.0, 2.0, 2.0]: [1.0, 2.0, 3.0] (Point A)
    # For [8.0, 8.0, 8.0]: [7.0, 8.0, 9.0] (Point C)
    # For [100.0, 100.0, 100.0]: [7.0, 8.0, 9.0] (Point C)
    # For [0.0, 0.0, 0.0]: [1.0, 2.0, 3.0] (Point A)
    # For [-1.0, -1.0, -1.0]: [1.0, 2.0, 3.0] (Point A)

print("\n")

################################################################################################
# 3. Function to search for nearest neighbor in KD-Tree
################################################################################################
print(
  "3. Function to search for nearest neighbor in KD-Tree"
)

class Node:
    def __init__(self, point=None, left=None, right=None, axis=None):
        self.point = point  # Store the point
        self.left = left    # Left child
        self.right = right  # Right child
        self.axis = axis    # Axis for splitting

def build_kd_tree(points, depth=0):
    n = len(points)  # Get the number of points
    if n == 0:
        return None  # Base case: no points left to create nodes

    axis = depth % 3  # Choose axis based on depth (0, 1, or 2)
    sorted_points = sorted(points, key=lambda point: point[axis])  # Sort points by the current axis
    median_index = n // 2  # Find the median index
    median_point = sorted_points[median_index]  # Select the median point as the root

    # Create a node for the median point
    node = Node(point=median_point, axis=axis)

    # Recursively build left and right subtrees
    left_points = sorted_points[:median_index]
    right_points = sorted_points[median_index + 1:]

    node.left = build_kd_tree(left_points, depth + 1)  # Build left subtree
    node.right = build_kd_tree(right_points, depth + 1)  # Build right subtree

    return node

kd_tree_root = build_kd_tree(points)  # Build the KD-Tree structure

def nearest_neighbor_search(root, target_point, current_best=None, depth=0):
    if root is None:
        return current_best  # Base case: no more nodes to search

    axis = depth % 3  # Determine the current axis
    current_point = root.point  # Get the point at the current node
    distance = euclidean_distance(target_point, current_point)  # Calculate distance to the current point

    # Initialize best distance based on current best
    best_distance = float('inf') if current_best is None else euclidean_distance(target_point, current_best.point)

    if distance < best_distance:  # If found a closer point
        current_best = root  # Update current best
        best_distance = distance  # Update best distance

    # Determine which branch to explore first
    if target_point[axis] < current_point[axis]:
        next_branch = root.left
        opposite_branch = root.right
    else:
        next_branch = root.right
        opposite_branch = root.left

    # Search the next branch
    current_best = nearest_neighbor_search(next_branch, target_point, current_best, depth + 1)

    # Backtracking condition: check if we need to search the opposite branch
    if abs(target_point[axis] - current_point[axis]) < best_distance:
        current_best = nearest_neighbor_search(opposite_branch, target_point, current_best, depth + 1)

    return current_best  # Return the best found neighbor

for target_point in target_points + extreme_target_points:
    nearest_point_kd = nearest_neighbor_search(kd_tree_root, target_point)  # Find nearest point using KD-Tree
    print(f"    Nearest point (KD-Tree Search) to {target_point}: {nearest_point_kd.point}")  # Actual value
    # Expected:
    # For [5.0, 5.0, 5.0]: [4.0, 5.0, 6.0] (Point B)
    # For [2.0, 2.0, 2.0]: [1.0, 2.0, 3.0] (Point A)
    # For [8.0, 8.0, 8.0]: [7.0, 8.0, 9.0] (Point C)
    # For [100.0, 100.0, 100.0]: [7.0, 8.0, 9.0] (Point C)
    # For [0.0, 0.0, 0.0]: [1.0, 2.0, 3.0] (Point A)
    # For [-1.0, -1.0, -1.0]: [1.0, 2.0, 3.0] (Point A)
