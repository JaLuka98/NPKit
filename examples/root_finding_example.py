from template_python_lib.roots.simple import newtons_method


# Define the function and its derivative
def f(x):
    return x**2 - 2


def f_prime(x):
    return 2 * x


# Initial guess
x0 = 1.0

# Find the root
root = newtons_method(f, f_prime, x0)
print(f"The root is approximately: {root}")
