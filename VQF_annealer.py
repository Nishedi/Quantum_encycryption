import math
import gurobipy as gp
from gurobipy import GRB


def factor_with_gurobi_bits(N: int, verbose: bool = True):
    if verbose:
        print(f" N = {N}")

    max_p = int(math.isqrt(N))
    max_q = N // 3
    if verbose:
        print(f"max_p = {max_p}, max_q = {max_q}")

    num_p_bits = max(1, math.ceil(math.log2((max_p - 1) / 2 + 1))) if max_p >= 3 else 1
    num_q_bits = max(1, math.ceil(math.log2((max_q - 1) / 2 + 1))) if max_q >= 3 else 1
    if verbose:
        print(f"num_p_bits = {num_p_bits}, num_q_bits = {num_q_bits}")

    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if verbose else 0)
        if not verbose:
            env.setParam("LogToConsole", 0)
        env.start()
        model = gp.Model("VQF_Binary_Factoring", env=env)
        model.Params.NonConvex = 2



        x = model.addVars(num_p_bits, vtype=GRB.BINARY, name="x")
        y = model.addVars(num_q_bits, vtype=GRB.BINARY, name="y")

        p_expr = 1 + gp.quicksum((2 ** i) * x[i - 1] for i in range(1, num_p_bits + 1)) # p = 1 + sum(2^i * x_i)
        q_expr = 1 + gp.quicksum((2 ** j) * y[j - 1] for j in range(1, num_q_bits + 1)) # q = 1 + sum(2^j * y_j)

        p_var = model.addVar(vtype=GRB.INTEGER, lb=3, ub=max_p, name="p") # Zmienna całkowita dla p
        q_var = model.addVar(vtype=GRB.INTEGER, lb=3, ub=max_q, name="q") # Zmienna całkowita dla q
        pq_var = model.addVar(vtype=GRB.INTEGER, lb=9, ub=N * 2, name="pq_product") # Zmienna całkowita dla p*q

        model.addConstr(p_var == p_expr, name="p_definition") #
        model.addConstr(q_var == q_expr, name="q_definition")
        model.addConstr(pq_var == p_var * q_var, name="quadratic_product") #
        model.addConstr(pq_var == N, name="exact_match")
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.IntegralityFocus = 1

        model.setObjective(0, GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            p_val = int(round(p_var.X))
            q_val = int(round(q_var.X))

            if verbose:
                print(f"p: {p_val}")
                print(f"q: {q_val}")
                if p_val * q_val == N:
                    print("Optimal - correct")
                else:
                    print("NonOptimal - incorrect")

            if verbose:
                x_bits = [int(round(x[i].X)) for i in range(num_p_bits)]
                y_bits = [int(round(y[j].X)) for j in range(num_q_bits)]
                x_bits.insert(0, 1)
                y_bits.insert(0, 1)
                n_bits = [1 if (N & (1 << i)) else 0 for i in range(N.bit_length())]
                print(f"p bits: (x1, x2...): {x_bits}, {p_val.bit_length()} bits")
                print(f"q bits: (y1, y2...): {y_bits}, {q_val.bit_length()} bits")
                print(f"N bits: (n1, n2...): {n_bits}, {N.bit_length()} bits")


            return p_val, q_val
        else:
            print("Error - No optimal solution found.")
            return None, None

    except gp.GurobiError as e:
        print(f"Gurobi error: {e.errno}: {e}")
        return None, None


def main():
    verbose=True
    p_tab_20 = [894643]
    p_tab_30 = [757871393]
    p_tab_40 = [629346195229]
    q_tab_20 = [613169]
    q_tab_30 = [1016990027]
    q_tab_40 = [694802246617]
    p_tab = p_tab_30
    q_tab = q_tab_30
    N=p_tab[0]*q_tab[0]

    p,q = factor_with_gurobi_bits(N=N, verbose=verbose)
    if p is not None and q is not None:
        if p*q != N:
            print(f"\n\nTest N = {N}", end=" ")
            print(f" - Incorrect: p={p}, q={q}, p*q={p*q}")
            print(f"{p_tab[j],q_tab[i]}")
        elif p*q == N and verbose:
            print(f"\n\nTest  N = {N}", end=" ")
            print(f" - Correct: p={p}, q={q}, p*q={p*q}")
            print(f"{p_tab[j],q_tab[i]}")
    else:
        print(f"\n\nTest N = {N}", end=" ")
        print(" - No optimal solution found.")



if __name__ == "__main__":
    main()