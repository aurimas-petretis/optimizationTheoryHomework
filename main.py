import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize, linprog

def cvxpy_example():
    import cvxpy as cp
    print('Example cvxpy')
    # Define variables
    x = cp.Variable()
    y = cp.Variable()

    # Define the objective function
    objective = cp.Minimize(x**2 + y**2)

    # Define constraints
    constraints = [x + y == 1, x - y >= 1]

    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print(f"Optimal value: {problem.value}")
    print(f"Optimal variable values: x = {x.value}, y = {y.value}")


def scipy_example():
    print('Example scipy')
    # Define the objective function
    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    # Define the constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 1}
    )

    # Initial guess
    x0 = [0, 0]

    # Solve the problem
    result = minimize(objective, x0, constraints=constraints)

    print(f"Optimal value: {result.fun}")
    print(f"Optimal variable values: x = {result.x[0]}, y = {result.x[1]}")


def scipy_example_playground():
    print('Example scipy playground')
    # Define the objective function
    def objective(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    # Define the constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - x[2] - 1}
    )

    # Initial guess
    x0 = [0, 0, 0]

    # Solve the problem
    result = minimize(objective, x0, constraints=constraints)

    print(f"Optimal value: {result.fun}")
    print(f"Optimal variable values: x = {result.x[0]}, y = {result.x[1]}")


def problem_1():
    print('Problem 1. Bounded optimization')

    x0 = 1
    def objective(x):
        return ((x[0] ** 2 - 3) ** 2) / 7 - 1

    sample_points = [x0]
    def callback(xk):
        sample_points.append(xk[0])

    result = minimize(objective, x0, bounds=[(0, 3)], callback=callback)

    print(f"f(x)={result.fun}, x={round(result.x[0], 4)}")
    plot_function(objective, sample_points, result, np.linspace(0, 3, 100))
    plot_function(objective, sample_points, result, np.linspace(0, 10, 100))


def problem_2():
    print('Problem 2. Unconstrained optimization')
    print("x0 | y0 | Number of steps | Number of function evaluations | min f(x,y) | x min | y min")
    problem_2_go([0, 0])
    problem_2_go([0.001, 0.001])
    problem_2_go([1, 1])
    problem_2_go([0.3, 0.7])


def problem_2_go(x0):
    def objective(x):
        return -x[0] * x[1] * (1 - x[0] * x[1]) / (x[0] + x[1])

    sample_points = [x0]
    def callback(xk):
        sample_points.append(xk)

    result = minimize(objective, x0, callback=callback)

    print(x0[0], x0[1], result.nit, result.nfev, round(result.fun, 4), round(result.x[0], 4), round(result.x[1], 4))
    plot_2D(objective, sample_points, result, np.linspace(0, 1.02, 100))


def problem_3():
    print('Problem 3. Constrained optimization')
    print("x0 | y0 | z0 | Number of steps | Number of function evaluations | min f(x,y) | x min | y min")
    problem_3_go([0, 0, 0])
    problem_3_go([0.001, 0.001, 0.001])
    problem_3_go([1, 1, 1])
    problem_3_go([0.6, 0.3, 0.7])


def problem_3_go(x0):
    def objective(x):
        return -x[0] * x[1] * x[2]

    constraints = (
        {'type': 'eq', 'fun': lambda x: x[0] * x[1] + x[0] * x[2] + x[1] * x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: x[0] * x[1] * x[2]}
    )

    sample_points = [x0]
    def callback(xk):
        sample_points.append(xk)

    result = minimize(objective, x0, constraints=constraints, callback=callback)

    print(x0[0], x0[1], x0[2], result.nit, result.nfev, round(result.fun, 4), round(result.x[0], 4), round(result.x[1], 4))


def problem_4():
    print('Problem 4. Penalty function optimization')
    print("x0 | y0 | z0 | Number of steps | Number of function evaluations | min f(x,y) | x min | y min")

    problem_4_go([0, 0, 0])
    problem_4_go([0.001, 0.001, 0.001])
    problem_4_go([1, 1, 1])
    problem_4_go([0.6, 0.3, 0.7])


def problem_4_go(x0):
    r = 0.01
    def objective(x):
        fx = -x[0] * x[1] * x[2]
        bx = abs(-x[0] * x[1] * x[2]) + abs(x[0] * x[1] + x[0] * x[2] + x[1] * x[2] - 1)
        return fx + (1 / r) * bx

    sample_points = [x0]
    def callback(xk):
        nonlocal r
        sample_points.append(xk)
        r *= 1.1

    result = minimize(objective, x0, callback=callback)

    print(x0[0], x0[1], x0[2], result.nit, result.nfev, round(result.fun, 4), round(result.x[0], 4), round(result.x[1], 4))


def problem_5():
    print('Problem 5. Linear optimization')
    print('maximize | f(x) | x')

    c = [2, -3, 0, -5]
    cmax = [-2, 3, 0, 5]
    A = [[-1, 1, -1, -1], [2, 4, 0, 0], [0, 0, 1, 1]]
    At = numpy.transpose(A)
    b1 = [8, 10, 3]
    b2 = [6, 3, 7]

    result1 = problem_5_go(c, A, b1, maximize=False)
    result1d = problem_5_go(b1, -At, c, maximize=False)
    result2 = problem_5_go(c, A, b2, maximize=False)
    result2d = problem_5_go(b2, -At, c, maximize=False)

    assert np.allclose((np.dot(A, result1.x) - b1) * result1d.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result1d.x) - c) * result1.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax
    assert np.allclose((np.dot(A, result2.x) - b2) * result2d.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result2d.x) - c) * result2.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax

    result1max = problem_5_go(cmax, A, b1, maximize=True)
    result1dmax = problem_5_go(b1, -At, cmax, maximize=True)
    result2max = problem_5_go(cmax, A, b2, maximize=True)
    result2dmax = problem_5_go(b2, -At, cmax, maximize=True)

    assert np.allclose((np.dot(A, result1max.x) - b1) * result1dmax.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result1dmax.x) - cmax) * result1max.x, [0, 0, 0]), "Problem 1 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax
    assert np.allclose((np.dot(A, result2max.x) - b2) * result2dmax.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied"
    # assert np.allclose((np.dot(At, result2dmax.x) - cmax) * result2max.x, [0, 0, 0]), "Problem 2 primal-dual problem condition not satisfied" # todo find a way to satisfy syntax


def problem_5_go(c, A, b, maximize=False):
    # bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)] # not needed, this is set by default
    result = linprog(c, A_ub=A, b_ub=b)
    print(maximize, round(-result.fun if maximize else result.fun, 4), result.x)
    return result


def plot_function(objective, sample_points, result, linspace):
    x_vals = linspace
    y_vals = [objective([x]) for x in x_vals]

    plt.plot(x_vals, y_vals, label="Objective function", color="blue")

    for idx, point in enumerate(sample_points):
        plt.scatter(point, objective([point]), color='orange', zorder=5, label=f"Sample point {idx}")

    plt.scatter(result.x[0], result.fun, color='red', zorder=5, label="Optimal point")

    plt.title("Objective Function and Optimization Result")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


def plot_functions(penalty_evaluation):
    for eval in penalty_evaluation:
        x_vals = eval[0]
        y_vals = eval[1]

        plt.plot(x_vals, y_vals, label="Objective function", color="blue")

    plt.title("Objective Function and Optimization Result")
    plt.xlabel("r")
    plt.ylabel("B(X, r)")
    plt.legend()
    plt.show()


# Define plot_2D function to create a contour plot
def plot_2D(objective, sample_points, result, linspace):
    # Generate a grid of points in the x and y range
    x_vals = linspace
    y_vals = linspace
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calculate the objective function values for the grid
    Z = np.array([objective([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Create the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)


    for idx, point in enumerate(sample_points):
        plt.scatter(point[0], point[1], color='orange', zorder=5, label=f"Sample point {idx}")

    # Mark the optimal point
    plt.scatter(result.x[0], result.x[1], color='red', s=100, label='Optimal point', zorder=10)

    # Customize the plot
    plt.title('Objective Function Contour and Optimization Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show()


def main():
    print('Starting the program...')
    # cvxpy_example()
    # scipy_example()
    scipy_example_playground()
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
    print('Program finished')


main()
