import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys

def shift(number, std_dev):
    shift = np.random.normal(0, std_dev)  # Generate Gaussian shift
    return number + shift

def line_count(filepath: str) -> int:
    """Counts lines efficiently using a generator expression."""
    try:
        with open(filepath, 'r') as file:
            return sum(1 for _ in file)  # The generator reads the file line by line
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return 0

def extract_numbers(filename):
    numbers = []  # Store all lines
    with open(filename, "r") as file:
        for line in file:
            numbers.append(list(map(float, line.strip().split(","))))  # Convert to float
    return numbers

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range")

    def normalize(self):
        norm = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector.")
        self.x /= norm
        self.y /= norm
        self.z /= norm
