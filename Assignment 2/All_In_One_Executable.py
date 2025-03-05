
#######################################
# User Control Area
#######################################

max_elements = 20 # Maximum number of elements to perform
resolution = 2  # Resolution

#######################################

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import math


# Calculate n1 and n2 based on the maximum number of elements and resolution
n1 = max_elements // resolution  # n1 represents the number of sets of points
n2 = resolution  # n2 directly assigned to resolution



def load_env_to_vectors(file_path):
    """
    Load environment data from a CSV file into a list of 3D points.

    Parameters:
    - file_path: The path to the CSV file.

    Returns:
    - A list of 3D points where each point is represented as a list of [x, y, z] coordinates.
    """
    df = pd.read_csv(file_path, skiprows=1,
                     header=None)  # Read the CSV file, skipping the first row and without headers
    # Create a list of points by combining x, y, z coordinates from the DataFrame
    ray_hit_vertices = [[float(x), float(y), float(z)] for x, y, z in zip(df[1], df[2], df[3])]
    return ray_hit_vertices  # Return the list of points


# Define the file path for the environment data
env_file_path = os.path.expanduser(r'TestEnv.csv')
env = load_env_to_vectors(env_file_path)  # Load environment data into vectors


# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
    - point1: The first point as a list of [x, y, z].
    - point2: The second point as a list of [x, y, z].

    Returns:
    - The Euclidean distance as a float.
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))  # Compute the distance


# Linear search function to find the nearest neighbor
def linear_search_nearest_neighbor(points, target):
    """
    Find the nearest neighbor to a target point using linear search.

    Parameters:
    - points: A list of points to search from.
    - target: The target point as a list of [x, y, z].

    Returns:
    - The nearest point found.
    """
    nearest = None  # Initialize nearest neighbor as None
    min_distance = float('inf')  # Initialize minimum distance to infinity

    for point in points:  # Iterate through each point
        distance = euclidean_distance(target, point)  # Calculate distance to the target
        if distance < min_distance:  # If this distance is smaller than the minimum found
            min_distance = distance  # Update minimum distance
            nearest = point  # Update nearest point

    return nearest  # Return the nearest point


# KD-Tree node class to represent each node in the KD-Tree
class Node:
    def __init__(self, point=None, left=None, right=None, axis=None):
        self.point = point  # The point stored in the node
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.axis = axis  # Axis along which the node is split


# Function to build a KD-Tree from a list of points
def build_kd_tree(points, depth=0):
    """
    Build a KD-Tree from a list of points.

    Parameters:
    - points: A list of points to construct the tree from.
    - depth: The current depth of the tree (used for choosing the axis).

    Returns:
    - The root node of the constructed KD-Tree.
    """
    n = len(points)  # Get the number of points
    if n == 0:  # Base case: If there are no points
        return None  # Return None

    axis = depth % 3  # Determine the axis to split on (0: x, 1: y, 2: z)
    sorted_points = sorted(points, key=lambda point: point[axis])  # Sort points based on the chosen axis
    median_index = n // 2  # Find the median index
    median_point = sorted_points[median_index]  # Select the median point

    node = Node(point=median_point, axis=axis)  # Create a new node with the median point

    # Recursively build left and right subtrees
    left_points = sorted_points[:median_index]  # Points on the left of the median
    right_points = sorted_points[median_index+1:]  # Points on the right of the median
    node.left = build_kd_tree(left_points, depth + 1)  # Build left subtree
    node.right = build_kd_tree(right_points, depth + 1)  # Build right subtree

    return node  # Return the root node of the KD-Tree


# Function to perform nearest neighbor search in a KD-Tree
def nearest_neighbor_search(root, target_point, current_best=None, depth=0):
    """
    Search for the nearest neighbor of a target point in a KD-Tree.

    Parameters:
    - root: The current node of the KD-Tree.
    - target_point: The target point to search for.
    - current_best: The current best point found (for backtracking).
    - depth: The current depth of the search (used for choosing the axis).

    Returns:
    - The nearest neighbor found in the KD-Tree.
    """
    if root is None:  # If the current node is None
        return current_best  # Return the current best point

    axis = depth % 3  # Determine the axis to compare
    current_point = root.point  # Get the point at the current node
    distance = euclidean_distance(target_point, current_point)  # Calculate distance to the current point

    # Determine the best distance found so far
    best_distance = float('inf') if current_best is None else euclidean_distance(target_point, current_best.point)


    # If the current point is closer than the current best, update the best point
    if distance < best_distance:
        current_best = root
        best_distance = distance  # Update the best distance

    # Decide which branch of the KD-Tree to explore next
    if target_point[axis] < current_point[axis]:
        next_branch = root.left  # Go left if target is less than current point
        opposite_branch = root.right  # Otherwise, opposite branch
    else:
        next_branch = root.right  # Go right if target is greater than current point
        opposite_branch = root.left

    # Recursively search in the next branch
    current_best = nearest_neighbor_search(next_branch, target_point, current_best, depth + 1)

    # Backtracking condition to check if we need to explore the opposite branch
    if abs(target_point[axis] - current_point[axis]) < best_distance:
        current_best = nearest_neighbor_search(opposite_branch, target_point, current_best, depth + 1)

    return current_best  # Return the nearest neighbor found

# Function to generate a list of random points
def generate_random_points(num_points):
    """
    Generate a list of random 3D points.

    Parameters:
    - num_points: The number of random points to generate.

    Returns:
    - A list of random 3D points.
    """
    return [(random.random(), random.random(), random.random()) for _ in range(num_points)]

# Build the KD-Tree
start_time = time.time()  # Start timing the KD-Tree building process
kd_tree_root = build_kd_tree(env)  # Build the KD-Tree using the environment points
kd_tree_build_time = time.time() - start_time  # Calculate the time taken to build the KD-Tree
print(f"KD-Tree build time: {kd_tree_build_time:.4f} seconds")  # Print the build time

# Prepare query screens with random points for searching
query_screens = []
for i in range(n1):
    query_screens.append(generate_random_points(n2 * (i + 1)))  # Generate increasing number of random points

# Measure the time for Linear Search
print("\nLinear Search Start-------------")
min_distance_linear = []  # List to hold minimum distances found
avg_distance_linear = []  # List to hold average distances for each task
N_list_linear = []  # List to hold the number of queries for Linear Search
time_list_linear = []  # List to hold time taken for each Linear Search

# Perform Linear Search for each screen of queries
for screen in query_screens:
    N = len(screen)  # Number of queries in the current screen

    start_time = time.time()  # Start timing the Linear Search
    for query in screen:  # For each query point in the current screen
        closest_point = linear_search_nearest_neighbor(env, query)  # Find closest point using linear search
        min_distance_linear.append(euclidean_distance(query, closest_point))  # Calculate and store the distance
    linear_search_time = time.time() - start_time  # Measure the total time taken for the search

    avg_distance_linear.append(np.mean(min_distance_linear))  # Calculate average distance for this screen
    min_distance_linear = []  # Reset the minimum distance list for the next screen

    N_list_linear.append(N)  # Store the number of queries
    time_list_linear.append(linear_search_time)  # Store the search time

    print(f"  Linear search time (for {len(screen)} queries): {linear_search_time:.4f} seconds")  # Print time taken for search


# Measure the time for KD-Tree Search
print("\nKD-Tree Search Start-------------")
min_distance_kd = []  # List to hold minimum distances found for KD-Tree
avg_distance_kd = []  # List to hold average distances for each task in KD-Tree search
N_list_kd = []  # List to hold the number of queries for KD-Tree Search
time_list_kd = []  # List to hold time taken for each KD-Tree Search

# Perform KD-Tree Search for each screen of queries
for screen in query_screens:
    N = len(screen)  # Number of queries in the current screen
    start_time = time.time()  # Start timing the KD-Tree Search
    for query in screen:  # For each query point in the current screen
        closest_point = nearest_neighbor_search(kd_tree_root, query).point  # Find closest point using KD-Tree search
        min_distance_kd.append(euclidean_distance(query, closest_point))  # Calculate and store the distance
    kd_tree_search_time = time.time() - start_time  # Measure the total time taken for the search

    avg_distance_kd.append(np.mean(min_distance_kd))  # Calculate average distance for this screen
    min_distance_kd = []  # Reset the minimum distance list for the next screen

    N_list_kd.append(N)  # Store the number of queries
    time_list_kd.append(kd_tree_search_time)  # Store the search time

    print(f"  KD-Tree search time (for {len(screen)} queries): {kd_tree_search_time:.4f} seconds")  # Print time taken for search

# Print the average distances found by both search methods
print("\nAverage Distance by Linear Search in Each Task: (Must be equal)")
avg_distance_linear = [float(x) for x in avg_distance_linear]  # Convert to float for consistency
print(avg_distance_linear)  # Print the average distances for Linear Search

print("\nAverage Distance by KD-Tree Search in Each Task: (Must be equal)")
avg_distance_kd = [float(x) for x in avg_distance_kd]  # Convert to float for consistency
print(avg_distance_kd)  # Print the average distances for KD-Tree Search

# Visualization of the search time results
plt.figure(figsize=(10, 6))  # Set the figure size for the plot
plt.plot(N_list_linear, time_list_linear, label='Linear Search', marker='o')  # Plot Linear Search times
plt.plot(N_list_kd, time_list_kd, label='KD-Tree Search', marker='o')  # Plot KD-Tree Search times
plt.xlabel('Number of Elements (N)')  # Label for x-axis
plt.ylabel('Search Time (seconds)')  # Label for y-axis
plt.title('Search Time Comparison: Linear vs KD-Tree')  # Title of the plot
plt.legend()  # Show legend
plt.grid()  # Show grid
plt.show()  # Display the plot
