import os
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from functools import lru_cache
import inquirer


# Compute Chebyshev distance between two permutations sigma and pi
def d_C(sigma, pi):
    n = len(sigma)
    distance = 0
    for i in range(n):
        distance = max(distance, abs(sigma[i] - pi[i]))
    return distance


# Compute Kendall tau-distance between two permutations sigma and pi
@lru_cache(maxsize=None)
def d_K(sigma, pi):
    n = len(sigma)
    permutations = [0] * n
    for i in range(n):
        permutations[i] = pi.index(sigma[i]) + 1

    # Merge sort to compute inversions
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_merge = merge(left, right)

        total_inversions = inv_left + inv_right + inv_merge
        return merged, total_inversions

    def merge(left, right):
        result = []
        inversions = 0
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                inversions += len(left) - i
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result, inversions

    _, inversions = merge_sort(permutations)
    return inversions


# Generate all subsets of a set S_n
def generate_all_subsets(S_n):
    n = len(S_n)
    return [[int(x) for x in bin(i)[2:].zfill(n)] for i in range(2**n)]


# Calculate the cardinality (number of elements) in a subset C
def cardinality(C):
    return sum(C)


# Solve the optimization problem for even d
def P_even(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix):
    model = gp.Model()
    model.Params.SolFiles = f"results/P({n}, {d}) {distanceMetric}/solution"
    x = {}
    
    for i in range(1, n_factorial + 1):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x{i}")

    model.setObjective(sum(x[i] for i in range(1, n_factorial + 1)), GRB.MAXIMIZE)

    for sigma in S_n:
        index = permutation_index_map[sigma]
        indices = [i for i, distance in enumerate(adjacency_matrix[index]) if distance <= d - 1]
        indices.remove(index)
        model.addConstr(x[index + 1] * sum(x[i + 1] for i in indices) == 0, f"c{index + 1}")

    model.optimize()
    model.write(f"results/P({n}, {d}) {distanceMetric}/model.mps")
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for i in range(1, n_factorial + 1):
            print(f"x{i} = {x[i].x}")
        print(f"Objective value = {model.objVal}")
    else:
        print("No feasible solution found")
    model.dispose()


# Solve the optimization problem for odd d
def P_odd(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix):
    t = (d - 1) // 2
    model = gp.Model()
    model.Params.SolFiles = f"results/P({n}, {d}) {distanceMetric}/solution"
    x = {}
    
    for i in range(1, n_factorial + 1):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x{i}")

    model.setObjective(sum(x[i] for i in range(1, n_factorial + 1)), GRB.MAXIMIZE)

    for sigma in S_n:
        index = permutation_index_map[sigma]
        indices = [i for i, distance in enumerate(adjacency_matrix[index]) if distance <= t]
        model.addConstr(sum(x[i + 1] for i in indices) <= 1, f"c{index + 1}")

    model.optimize()
    model.write(f"results/P({n}, {d}) {distanceMetric}/.mps")
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for i in range(1, n_factorial + 1):
            print(f"x{i} = {x[i].x}")
        print(f"Objective value = {model.objVal}")
    else:
        print("No feasible solution found")
    model.dispose()


# Calculate the size of the largest subset of permutations with min Kendall tau-distance d
def P(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix):
    if d % 2 == 0:
        P_even(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix)
    else:
        P_odd(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix)
    # Or, we can use the P_even formulation for odd d as well
    # P_even(n, d, n_factorial, S_n, permutation_index_map, adjacency_matrix)


if __name__ == "__main__":
    questions = [
        inquirer.Text("n", message="Enter n", validate=lambda _, x: int(x) >= 2),
        inquirer.Text("d", message="Enter d", validate=lambda _, x: int(x) >= 1),
        inquirer.List("distanceMetric", message="Choose a distance metric", choices=["Chebyshev", "Kendall Tau"], default="Chebyshev")
    ]
    answers = inquirer.prompt(questions)
    n = int(answers["n"])
    d = int(answers["d"])
    distanceMetric = answers["distanceMetric"]
    
    S_n = list(permutations(range(1, n + 1)))
    n_factorial = len(S_n)
    permutation_index_map = {permutation: i for i, permutation in enumerate(S_n)}

    adjacency_matrix = []
    for sigma in S_n:
        row = []
        for pi in S_n:
            if distanceMetric == "Chebyshev":
                row.append(d_C(sigma, pi))
            elif distanceMetric == "Kendall Tau":
                row.append(d_K(sigma, pi))
            else:
                raise ValueError("Invalid distance metric")
        adjacency_matrix.append(row)

    # Create directory for storing results if it does not exist
    directory = f"results/P({n}, {d}) {distanceMetric}"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    P(n, d, distanceMetric, n_factorial, S_n, permutation_index_map, adjacency_matrix)
    