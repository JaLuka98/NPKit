from template_python_lib.roots.simple import newtons_method


def test_newtons_method():
    # Define the function and its derivative
    def f(x):
        return x**2 - 4  # Root at x = ±2

    def f_prime(x):
        return 2 * x

    # Initial guess
    x0 = 1.0

    # Call the Newton's method
    root = newtons_method(f, f_prime, x0)

    # Assert that the root is approximately 2
    assert abs(root - 2) < 1e-7, f"Expected root near 2, but got {root}"
