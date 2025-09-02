def newtons_method(func, func_derivative, x0, tolerance=1e-7, max_iterations=100):
    """
    Find the root of a function using Newton's method.

    Parameters:
        func (callable): The function for which the root is to be found.
        func_derivative (callable): The derivative of the function.
        x0 (float): Initial guess for the root.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        float: The estimated root.
    """
    x = x0
    for i in range(max_iterations):
        f_x = func(x)
        f_prime_x = func_derivative(x)

        if abs(f_x) < tolerance:
            return x

        if f_prime_x == 0:
            raise ValueError("Derivative is zero. No solution found.")

        x = x - f_x / f_prime_x

    raise ValueError("Maximum iterations reached. No solution found.")
