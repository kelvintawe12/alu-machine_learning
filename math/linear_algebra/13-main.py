#!/usr/bin/env python3

import numpy as np
np_cat = __import__('13-cats_got_your_tongue').np_cat

# Original tests
mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])
mat3 = np.array([[7], [8]])
print("2D int arrays, axis=0:")
print(np_cat(mat1, mat2))
print("2D int arrays, axis=1:")
print(np_cat(mat1, mat2, axis=1))
print("2D int arrays, axis=1 with different shapes:")
print(np_cat(mat1, mat3, axis=1))

# Higher dimensional arrays
print("\n3D float arrays, axis=0:")
mat1_3d = np.random.rand(2, 3, 4)
mat2_3d = np.random.rand(1, 3, 4)
print("Shape mat1:", mat1_3d.shape)
print("Shape mat2:", mat2_3d.shape)
result_3d = np_cat(mat1_3d, mat2_3d, axis=0)
print("Result shape:", result_3d.shape)

print("\n3D float arrays, axis=2:")
mat2_3d_axis2 = np.random.rand(2, 3, 2)  # Match shape except axis=2
result_3d_axis2 = np_cat(mat1_3d, mat2_3d_axis2, axis=2)
print("Shape mat1:", mat1_3d.shape)
print("Shape mat2:", mat2_3d_axis2.shape)
print("Result shape axis=2:", result_3d_axis2.shape)

# Different data types
print("\nFloat arrays:")
mat1_float = np.array([[1.1, 2.2], [3.3, 4.4]])
mat2_float = np.array([[5.5, 6.6], [7.7, 8.8]])
print("Float concat axis=0:")
print(np_cat(mat1_float, mat2_float))

# Edge case: axis out of bounds (should raise error)
try:
    print("\nTrying axis=3 on 3D array:")
    np_cat(mat1_3d, mat2_3d, axis=3)
except Exception as e:
    print("Error:", e)
