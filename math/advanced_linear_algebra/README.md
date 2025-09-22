# Advanced Linear Algebra

This directory contains implementations of advanced linear algebra concepts in Python 3.

## Files

- `0-determinant.py` - Function to calculate the determinant of a matrix
- `1-minor.py` - Function to calculate the minor matrix of a matrix
- `2-cofactor.py` - Function to calculate the cofactor matrix of a matrix
- `3-adjugate.py` - Function to calculate the adjugate matrix of a matrix
- `4-inverse.py` - Function to calculate the inverse of a matrix
- `5-definiteness.py` - Function to determine the definiteness of a matrix
- `0-main.py` through `5-main.py` - Test files for each function
- `__init__.py` - Package initialization file

## Functions

### determinant(matrix)
Calculates the determinant of a square matrix using recursive expansion by minors.

**Args:**
- matrix (list): A list of lists representing a square matrix

**Returns:**
- float: The determinant of the matrix

**Raises:**
- TypeError: If matrix is not a list of lists
- ValueError: If matrix is not square

### minor(matrix)
Calculates the minor matrix by computing the determinant of each submatrix.

**Args:**
- matrix (list): A list of lists representing a square matrix

**Returns:**
- list: The minor matrix

**Raises:**
- TypeError: If matrix is not a list of lists
- ValueError: If matrix is not square or is empty

### cofactor(matrix)
Calculates the cofactor matrix by applying the (-1)^(i+j) sign pattern to the minor matrix.

**Args:**
- matrix (list): A list of lists representing a square matrix

**Returns:**
- list: The cofactor matrix

**Raises:**
- TypeError: If matrix is not a list of lists
- ValueError: If matrix is not square or is empty

### adjugate(matrix)
Calculates the adjugate (adjoint) matrix by transposing the cofactor matrix.

**Args:**
- matrix (list): A list of lists representing a square matrix

**Returns:**
- list: The adjugate matrix

**Raises:**
- TypeError: If matrix is not a list of lists
- ValueError: If matrix is not square or is empty

### inverse(matrix)
Calculates the inverse of a matrix using the formula: (1/det(A)) * adj(A).

**Args:**
- matrix (list): A list of lists representing a square matrix

**Returns:**
- list: The inverse matrix, or None if matrix is singular

**Raises:**
- TypeError: If matrix is not a list of lists
- ValueError: If matrix is not square or is empty

### definiteness(matrix)
Determines the definiteness of a matrix by analyzing its eigenvalues.

**Args:**
- matrix (numpy.ndarray): A numpy array representing a square matrix

**Returns:**
- str: One of: "Positive definite", "Positive semi-definite", "Negative definite", "Negative semi-definite", "Indefinite", or None

**Raises:**
- TypeError: If matrix is not a numpy.ndarray

## Requirements

- Python 3.5+
- numpy 1.15 (for definiteness function only)
- All files must be executable
- Code follows pycodestyle (version 2.5) guidelines

## Usage

Run the test files to verify functionality:

```bash
./0-main.py  # Test determinant function
./1-main.py  # Test minor function
./2-main.py  # Test cofactor function
./3-main.py  # Test adjugate function
./4-main.py  # Test inverse function
./5-main.py  # Test definiteness function
```

## Learning Objectives

After completing this project, you should be able to explain:

- What a determinant is and how to calculate it
- What a minor, cofactor, and adjugate are and how to calculate them
- What a matrix inverse is and how to calculate it
- What eigenvalues and eigenvectors are and how to calculate them
- What matrix definiteness is and how to determine it
