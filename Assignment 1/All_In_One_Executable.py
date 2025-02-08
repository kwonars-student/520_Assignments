import math
from bitarray import bitarray
from cuckoofilter import CuckooFilter
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Disclaimer: Most of the code has been modified based on results obtained from Generative AI.
# Sources are cited where applicable.

'''
1. The code is structured into two main parts: 
   a. Definition of Functions and Classes 
   b. Execution for testing purposes.

2. Following originals:
    When setting filter parameters (e.g., target error rate, capacity) for filters, the intentions of Bloom et al. (1970) and Fan et al. (2014) were considered.

3. Special features:
    A Truncate Filter was developed, which outperforms linear search in terms of performance.

4. Minimize throughput limitation:
    Efforts were made to minimize elements that could impact time complexity due to memory input/output operations.

5. Easy test without datasets:
    Unit test available. The code was designed to be lightweight and intuitive, eliminating the need for saving and loading additional datasets.

6. Smooth visualization:
    In visualizations through graphs, a Butterworth Filter was applied to ensure clear trend recognition.
'''

###############################################
###### 1. Function and Class Definitions ######
###############################################


###### 1.1 Function #######


# A simple hash function with a set seed for fair comparison.
# This hash is applied to all available filters

def simple_hash(number, seed):
    # Use bitwise XOR to combine the seed and number
    # XOR operation ensures that the resulting hash changes with the input values
    return (seed ^ number) * 0x9e3779b9 & 0xFFFFFFFF  # Golden ratio constant for better distribution, mask to 32-bit


# A sampling function to extract login IDs that are unique and pre-sorted
def sample_from_items(items, num_samples):
    n = len(items)

    if num_samples > n:
        raise ValueError("The number of samples cannot exceed the length of the list.")

    # Calculate the maximum gap between each sample (total number of elements / number of samples)
    max_gap = n // num_samples

    # Select the first sample randomly from the range 0 to num_samples-1
    current_index = random.randint(0, num_samples - 1)
    samples = [items[current_index]]

    # After that, maintain the gap and select subsequent samples
    for _ in range(num_samples - 1):
        # Calculate the maximum range for selecting the next sample (it should not exceed current_index + max_gap)
        next_index_max = min(current_index + max_gap, n - 1)

        # If the range is not empty, select a sample
        if next_index_max > current_index:
            next_index = random.randint(current_index + 1, next_index_max)
            samples.append(items[next_index])
            current_index = next_index  # Update the current index
        else:
            break  # Stop if no more samples can be selected

    return samples

# Search and Check Implementation
def linear_search(items, target):
    for item in items:
        if item == target:
            return True
    return False

def binary_search(items, target):
    left, right = 0, len(items) - 1
    while left <= right:
        mid = (left + right) // 2
        if items[mid] == target:
            return True
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

def measure_linear_search(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):
        sample_items = random.sample(items, n)

        start_time = time.time()
        for item in sample_items:
            linear_search(sample_items, item)
        linear_check_times.append((time.time() - start_time))

        # Print progress (only update recent values)
        percent_done = (i + 1) / total_steps * 100
        print(f"\rLinear Search Progress: {percent_done:.2f}%", end="")


def measure_binary_search(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):

        sample_items = items[:n]

        # Measure binary search check performance using BinarySearchTree
        start_time = time.time()
        for item in sample_items:
            binary_search(sample_items, item)
        binary_check_times.append((time.time() - start_time))

        # Current progress
        percent_done = (i + 1) / total_steps * 100
        print(f"\rBinary Search Progress: {percent_done:.2f}%", end="")
def measure_hash_table(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):
        sample_items = random.sample(items, n)

        hash_table = HashTable(size=(int)(len(items)/2))
        for item in sample_items:
            hash_table.insert(item)

        # Hash Table Search Performance Test
        start_time = time.time()
        for item in sample_items:
            hash_table.exists(item)
        hash_check_times.append((time.time() - start_time))

        # Current progress
        percent_done = (i + 1) / total_steps * 100
        print(f"\rHash Table Progress: {percent_done:.2f}%", end="")

def measure_bloom_filter(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):
        # sample_items = items[:n]
        sample_items = random.sample(items, n)

        bloom_filter = BloomFilter(items_count=n, fp_prob=0.001) # Item count is expected to be same as the number of IDs
        for item in sample_items:
            bloom_filter.add(item)

        # Bloom Filter Search Performance
        start_time = time.time()
        for item in sample_items:
            bloom_filter.check(item)
        bloom_check_times.append((time.time() - start_time))

        # Current Progress
        percent_done = (i + 1) / total_steps * 100
        print(f"\rBloom Filter Progress: {percent_done:.2f}%", end="")

def measure_cuckoo_filter(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))  # Total number of steps to complete
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):
        # Randomly sample 'n' items from the list
        sample_items = random.sample(items, n)

        # Measure the insertion performance of the Cuckoo Filter
        cuckoo_filter = CuckooFilter(capacity=2*n, fingerprint_size=13)  # Parameters from Fan et al.'s paper
        # start_time = time.time()
        for item in sample_items:
            cuckoo_filter.insert(item)
        # cuckoo_insert_times.append(time.time() - start_time)

        # Measure the lookup performance of the Cuckoo Filter
        start_time = time.time()
        # random.shuffle(sample_items)
        for item in sample_items:
            cuckoo_filter.contains(item)
        cuckoo_check_times.append((time.time() - start_time))

        # Print the progress (only update the most recent value)
        percent_done = (i + 1) / total_steps * 100
        print(f"\rCuckoo Filter Progress: {percent_done:.2f}%", end="")

def measure_truncate_filter(items, step):
    total_steps = len(np.linspace(100, len(items), num=step, dtype=int))  # Total number of steps to complete
    for i, n in enumerate(np.linspace(100, len(items), num=step, dtype=int)):
        # Randomly sample 'n' items from the list
        sample_items = random.sample(items, n)

        # Measure the insertion performance of the Truncate Filter
        truncate_filter = TruncateFilter(bit_length=10)
        # start_time = time.time()
        for item in sample_items:
            truncate_filter.insert(item)
        # truncate_insert_times.append(time.time() - start_time)

        # Measure the lookup performance of the Truncate Filter
        start_time = time.time()
        # random.shuffle(sample_items)
        for item in sample_items:
            truncate_filter.contains(item)
        truncate_check_times.append((time.time() - start_time))

        # Print the progress (only update the most recent value)
        percent_done = (i + 1) / total_steps * 100
        print(f"\rTruncate Filter Progress: {percent_done:.2f}%", end="")

# Smoothing Function for Visualization
def butter_filter(data, cutoff=0.1, fs=1.0, order=2):
    b, a = butter(order, cutoff, btype='low', fs=fs)
    return filtfilt(b, a, data)


# Function to run the measurement and smoothing
def unit_run(method_key, maximum_N, resolution):
    methods = {
        1: ("Linear Search", measure_linear_search, linear_check_times, 'purple'),
        2: ("Binary Search", measure_binary_search, binary_check_times, 'orange'),
        3: ("Hash Table Check", measure_hash_table, hash_check_times, 'r'),
        4: ("Bloom Filter Check", measure_bloom_filter, bloom_check_times, 'g'),
        5: ("Cuckoo Filter Check", measure_cuckoo_filter, cuckoo_check_times, 'b'),
        6: ("Truncate Filter Check", measure_truncate_filter, truncate_check_times, 'brown'),
        "Linear Search": ("Linear Search", measure_linear_search, linear_check_times, 'purple'),
        "Binary Search": ("Binary Search", measure_binary_search, binary_check_times, 'orange'),
        "Hash Table": ("Hash Table Check", measure_hash_table, hash_check_times, 'r'),
        "Bloom Filter": ("Bloom Filter Check", measure_bloom_filter, bloom_check_times, 'g'),
        "Cuckoo Filter": ("Cuckoo Filter Check", measure_cuckoo_filter, cuckoo_check_times, 'b'),
        "Truncate Filter": ("Truncate Filter Check", measure_truncate_filter, truncate_check_times, 'brown')
    }

    if method_key not in methods:
        raise ValueError(f"Invalid method_key: {method_key}")

    label, measure_func, check_times, color = methods[method_key]

    # Sample IDs from the range, ensuring they are unique and sorted
    items = sample_from_items(range(login_id_min, login_id_max + 1), maximum_N)
    measure_func(items, resolution)  # Run the selected method's measurement function
    smoothed = butter_filter(check_times)  # Apply the Butterworth filter
    plt.plot(np.linspace(100, len(items), num=resolution, dtype=int), smoothed, label=label, color=color)


######## 1.2 Class #########

# Hash Table Implementation
class HashTable:
    def __init__(self, size=10000):
        # Initialize a hash table with the given size
        self.table = [False] * size  # The table is initialized with Boolean values
        self.size = size

    def _hash(self, key):
        # aVery simple hash function: tkes the number and calculates the remainder when divided by the table size
        return key % self.size

    def insert(self, key):
        # Insert a number into the hash table (set it to True)
        index = self._hash(key)
        self.table[index] = True

    def exists(self, key):
        # Check if the given number exists in the hash table
        index = self._hash(key)
        return self.table[index]

# Bloom Filter Implementation
class BloomFilter(object):

    '''
    Class for Bloom filter, using murmur3 hash function
    '''

    def __init__(self, items_count, fp_prob):
        '''
        items_count : int
            Number of items expected to be stored in bloom filter
        fp_prob : float
            False Positive probability in decimal
        '''
        # False possible probability in decimal
        self.fp_prob = fp_prob

        # Size of bit array to use
        self.size = self.get_size(items_count, fp_prob)

        # number of hash functions to use
        self.hash_count = self.get_hash_count(self.size, items_count)

        # Bit array of given size
        self.bit_array = bitarray(self.size)

        # initialize all bits as 0
        self.bit_array.setall(0)

    def add(self, item):
        '''
        Add an item in the filter
        '''
        digests = []
        for i in range(self.hash_count):

            # create digest for given item.
            # i work as seed to mmh3.hash() function
            # With different seed, digest created is different
            digest = simple_hash(item, i) % self.size
            digests.append(digest)

            # set the bit True in bit_array
            self.bit_array[digest] = True
    def check(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.hash_count):
            digest = simple_hash(item, i) % self.size
            if self.bit_array[digest] == False:

                # if any of bit is False then,its not present
                # in filter
                # else there is probability that it exist
                return False
        return True

    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p))/(math.log(2)**2)
        return int(m)

    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) * lg(2)

        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m/n) * math.log(2)
        return int(k)

# Custom filter: Truncate Filter Implementation
class TruncateFilter:
    def __init__(self, bit_length):
        self.bit_length = bit_length  # The bit length to truncate to
        # boolean_array is initialized with False, with the size of bit_length
        self.boolean_array = np.full(bit_length, False)

    def _hash(self, key):
        """Converts an integer to binary and truncates it to the bit_length from the end"""
        binary_str = bin(key)[2:]  # bin() includes '0b', so slice it off with [2:]
        truncated_binary = binary_str[-self.bit_length:].zfill(
            self.bit_length)  # Truncate from the end to bit_length and pad with 0 if needed
        # print(f"Truncated binary: {truncated_binary}")  # Output truncated_binary
        return truncated_binary

    def insert(self, key):
        """Converts the binary string to a boolean array and performs an OR operation"""
        # Convert binary string to a boolean array
        binary_array = np.array([int(bit) for bit in self._hash(key)], dtype=bool)

        # Perform OR operation
        result = np.logical_or(binary_array, self.boolean_array[:self.bit_length])  # OR with the size of boolean_array
        return result

    def contains(self, key):
        """Performs an AND operation and compares if it matches the truncated_binary"""
        # Convert binary string to a boolean array
        binary_array = np.array([int(bit) for bit in self._hash(key)], dtype=bool)

        # Perform AND operation
        and_result = np.logical_and(binary_array, self.boolean_array[:self.bit_length])

        # If the AND result matches the truncated_binary, return True, otherwise False
        return np.array_equal(and_result, binary_array)



##################################
##### 2. Execution for Test ######
##################################


##### 2.1 Settings ######

# Lists to store the y-values of the graph.
linear_check_times = []
binary_check_times = []
hash_check_times = []
bloom_check_times = []
cuckoo_check_times = []
truncate_check_times = []


# Parameters for comparison experiment
login_id_min = 1000000000
login_id_max = 2000000000  # A very large number to ensure sufficient range for sampling IDs


##### 2.2 Visualization and Smoothing Logic ######

# Plotting graphs
plt.figure(figsize=(12, 8))

##### 2.3 Test Units ######
maximum_N = 2000  # Number of items (IDs) to sample for the experiment
resolution = 20  # Step size for the experiment

unit_run("Linear Search", maximum_N, resolution)
unit_run("Binary Search", maximum_N, resolution)
unit_run("Hash Table", maximum_N, resolution)
unit_run("Bloom Filter", maximum_N, resolution)
unit_run("Cuckoo Filter", maximum_N, resolution)
unit_run("Truncate Filter", maximum_N, resolution)

# Label and Title
plt.xlabel('Number of Items')
plt.ylabel('Check Time')
plt.title('Comparison of Search Algorithms with Smoothed Data (Butterworth Filter)')
plt.legend()
plt.show()