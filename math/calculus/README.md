# Calculus

This directory contains solutions to various calculus problems covering:

## Topics Covered

### 1. Summation and Product Notation
- **Sigma Notation (Σ)**: Used for summation
- **Pi Notation (Π)**: Used for products
- Understanding series and their calculations

### 2. Derivatives
- Basic differentiation rules
- Product rule, chain rule
- Logarithmic differentiation
- Partial derivatives
- Mixed partial derivatives

### 3. Integrals
- Indefinite integrals
- Definite integrals
- Double integrals
- Polynomial integration

## Files

### Multiple Choice Questions (0-8, 11-16)
- `0-sigma_is_for_sum`: ∑_{i=2}^{5} i
- `1-seegma`: ∑_{k=1}^{4} 9i - 2k
- `2-pi_is_for_product`: ∏_{i=1}^{m} i
- `3-pee`: ∏_{i=0}^{10} i
- `4-hello_derivatives`: d/dx of x^4 + 3x^3 - 5x + 1
- `5-log_on_fire`: d/dx of xln(x)
- `6-voltaire`: d/dx of ln(x^2)
- `7-partial_truths`: ∂f/∂y where f(x,y) = e^{xy}
- `8-all-together`: ∂²/∂y∂x of e^{x^2y}
- `11-integral`: ∫3x^2 dx
- `12-integral`: ∫e^y dy
- `13-definite`: ∫_0^3 x^2 dx
- `14-definite`: ∫_{-1}^0 x dx
- `15-definite`: ∫_0^5 x dx
- `16-double`: ∫_1^2 3x^2 dx

### Python Scripts
- `9-sum_total.py`: Function to calculate ∑_{i=1}^{n} i^2
- `10-matisse.py`: Function to calculate polynomial derivatives
- `17-integrate.py`: Function to calculate polynomial integrals

### Test Files
- `9-main.py`: Test script for summation function
- `10-main.py`: Test script for derivative function
- `17-main.py`: Test script for integral function

## Usage

Each Python script can be run independently:

```bash
./9-main.py  # Should output: 55
./10-main.py # Should output: [3, 0, 3]
./17-main.py # Should output: [0, 5, 1.5, 0, 0.25]
```

## Mathematical Concepts Demonstrated

1. **Summation Formula**: ∑_{i=1}^{n} i^2 = n(n+1)(2n+1)/6
2. **Product Notation**: Factorial calculations
3. **Differentiation Rules**: Power rule, chain rule, product rule
4. **Integration Rules**: Basic integration, definite integrals
5. **Partial Derivatives**: Functions of multiple variables

## Learning Objectives

By the end of this project, you should be able to explain:
- Summation and product notation
- Series and common series
- Derivatives and differentiation rules
- Partial derivatives
- Indefinite and definite integrals
- Double integrals
