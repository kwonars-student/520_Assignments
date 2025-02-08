# Code Execution Manual

This document explains how to run experiments using different search and filter algorithms in a given setup. The experiments utilize different data structures like Linear Search, Binary Search, Hash Table, Bloom Filter, Cuckoo Filter, and Truncate Filter. We will also cover examples for customizing parameters, running specific functions, and commenting out parts of the code to test certain configurations.

## Table of Contents

1. [Introduction](#introduction)
2. [Running the Full Experiment](#running-the-full-experiment)
3. [Commenting Out Specific Algorithms](#commenting-out-specific-algorithms)
4. [Customizing Parameters](#customizing-parameters)
5. [Notes on the Code](#notes-on-the-code)

---

## Introduction

In this experiment, we run several different algorithms on a set of items to analyze their performance. These algorithms include:

- **Linear Search**
- **Binary Search**
- **Hash Table**
- **Bloom Filter**
- **Cuckoo Filter**
- **Truncate Filter**

The main parameters used for this experiment are `maximum_N` (the number of items to sample) and `resolution` (step size for the experiment).

---

## Running the Full Experiment

To run the full experiment, simply call the `unit_run` function for each algorithm. This will perform the experiment with default parameters:

```python
maximum_N = 2000  # Number of items (IDs) to sample for the experiment
resolution = 20  # Step size for the experiment

unit_run("Linear Search", maximum_N, resolution)
unit_run("Binary Search", maximum_N, resolution)
unit_run("Hash Table", maximum_N, resolution)
unit_run("Bloom Filter", maximum_N, resolution)
unit_run("Cuckoo Filter", maximum_N, resolution)
unit_run("Truncate Filter", maximum_N, resolution)
```
This will run all the algorithms on a set of 2000 items, stepping through the experiment with a resolution of 20.


---

## Commenting Out Specific Algorithms

If you want to run only a subset of the algorithms, you can comment out the lines that correspond to the algorithms you do not want to include in the experiment. 

For example, to run only the **Linear Search** and **Binary Search** algorithms, you can comment out the others like this:

```python
maximum_N = 2000  # Number of items (IDs) to sample for the experiment
resolution = 20  # Step size for the experiment

unit_run("Linear Search", maximum_N, resolution)
unit_run("Binary Search", maximum_N, resolution)
# unit_run("Hash Table", maximum_N, resolution)
# unit_run("Bloom Filter", maximum_N, resolution)
# unit_run("Cuckoo Filter", maximum_N, resolution)
# unit_run("Truncate Filter", maximum_N, resolution)
```
In this case, only the Linear Search and Binary Search algorithms will be executed, and the rest will be skipped.

This method can be helpful when you want to test specific algorithms individually or reduce the time needed to run experiments by excluding certain algorithms.
---

## Customizing Parameters

You can customize the experiment parameters to fit your specific needs by modifying the values of `maximum_N` and `resolution`.

- `maximum_N` controls the number of items (IDs) to sample for the experiment.
- `resolution` controls the step size for the experiment.

For example, if you want to run the experiment on only 1000 items and use a smaller `resolution` of 10, you can adjust the parameters like this:

```python
maximum_N = 1000  # Number of items (IDs) to sample for the experiment
resolution = 10   # Step size for the experiment

unit_run("Linear Search", maximum_N, resolution)
unit_run("Binary Search", maximum_N, resolution)
unit_run("Hash Table", maximum_N, resolution)
unit_run("Bloom Filter", maximum_N, resolution)
unit_run("Cuckoo Filter", maximum_N, resolution)
unit_run("Truncate Filter", maximum_N, resolution)
```
In this case, all the algorithms will run on 1000 items with a resolution of 10.

You can experiment with different values of `maximum_N` and `resolution` to observe how they affect the performance of each algorithm.
