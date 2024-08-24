import gurobipy as gp
from gurobipy import GRB
import numpy as np


def pip_fixed_problem(model, obj_cons_num, X, y, w_bar, b_bar, w_start, b_start, z_plus_start, z_minus_start, epsilon,
                      delta_1,
                      delta_2, gamma_0,
                      M, rho, beta_p, lbd, lbd_2, iterations, outer_or_fixed_iteration, dirname, full_mip, fixed):
    if not fixed:
        iter_name = '/outer_iter='
    else:
        iter_name = '/fixed_iter='
    model = model.copy()
    N = X.shape[0]
    dim = X.shape[1]
    item_plus = [N, N]
    item_minus = [0, N]
    z_plus = {}
    optimal_z_plus = {}
    z_minus = {}
    optimal_z_minus = {}
    J_0_plus, J_ge_plus, J_le_plus, J_0_minus, J_ge_minus, J_le_minus = {}, {}, {}, {}, {}, {}
    with open(dirname + '/Solution' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(
            iterations) + '.txt',
              'a') as f:
        print("delta_1,delta_2,epsilon:", delta_1, delta_2, epsilon, file=f)
    w = model.addVars(dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
    w_abs = model.addVars(dim, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w_abs")
    u = model.addVars(dim, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
    b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    gamma = model.addVar(lb=0, ub=gamma_0, vtype=GRB.CONTINUOUS, name="gamma")
    for p in range(dim):
        w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
        model.addConstr(w[p] <= w_abs[p])
        model.addConstr(-w[p] <= w_abs[p])
        model.addConstr(w_abs[p] <= u[p])
    b.setAttr(gp.GRB.Attr.Start, b_start)

    for i in range(obj_cons_num):
        z_plus[i] = {}
        z_minus[i] = {}
        for j in range(item_plus[i]):
            z_plus[i][j] = model.addVar(vtype=GRB.BINARY, name="z_plus_" + str(i) + "_" + str(j))
        for j in range(item_minus[i]):
            z_minus[i][j] = model.addVar(vtype=GRB.BINARY, name="z_minus_" + str(i) + "_" + str(j))

    for i in range(obj_cons_num):
        J_0_plus[i], J_ge_plus[i], J_le_plus[i], J_0_minus[i], J_ge_minus[i], J_le_minus[i] = [], [], [], [], [], []
        for j in range(item_plus[i]):
            if i == 0:
                if y[j] * (np.dot(w_start, X[j]) + b_start) >= delta_1:
                    J_ge_plus[i].append(j)
                    model.remove(z_plus[i][j])
                    model.addConstr(y[j] * (gp.quicksum(w[p] * X[j][p] for p in range(dim)) + b) >= 0)
                elif y[j] * (np.dot(w_start, X[j]) + b_start) <= -delta_2:
                    J_le_plus[i].append(j)
                    model.remove(z_plus[i][j])
                else:
                    J_0_plus[i].append(j)
                    z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                    model.addConstr(
                        y[j] * (gp.quicksum(w[p] * X[j][p] for p in range(dim)) + b) >= -M * (1 - z_plus[i][j]))
            if i == 1:
                if min(y[j], np.dot(w_start, X[j]) + b_start) >= delta_1:
                    J_ge_plus[i].append(j)
                    model.remove(z_plus[i][j])
                    model.addConstr(gp.quicksum(w[p] * X[j][p] for p in range(dim)) + b >= 0)
                elif min(y[j], np.dot(w_start, X[j]) + b_start) <= -delta_2:
                    J_le_plus[i].append(j)
                    model.remove(z_plus[i][j])
                else:
                    J_0_plus[i].append(j)
                    z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                    model.addConstr(-M * (1 - z_plus[i][j]) <= y[j])
                    model.addConstr(gp.quicksum(w[p] * X[j][p] for p in range(dim)) + b >= -M * (1 - z_plus[i][j]))
        for j in range(item_minus[i]):
            if i == 1:
                if -(np.dot(w_start, X[j]) + b_start) - epsilon >= delta_1:
                    J_ge_minus[i].append(j)
                    model.remove(z_minus[i][j])
                    model.addConstr(-gp.quicksum(w[p] * X[j][p] for p in range(dim)) - b >= epsilon)
                elif -(np.dot(w_start, X[j]) + b_start) - epsilon <= -delta_2:
                    J_le_minus[i].append(j)
                    model.remove(z_minus[i][j])
                else:
                    J_0_minus[i].append(j)
                    z_minus[i][j].setAttr(gp.GRB.Attr.Start, z_minus_start[i][j])
                    model.addConstr(
                        -gp.quicksum(w[p] * X[j][p] for p in range(dim)) - b >= -M * (1 - z_minus[i][j]) + epsilon)
    model.addConstr(gp.quicksum(z_plus[1][j] for j in J_0_plus[1]) + sum(1 for _ in J_ge_plus[1]) + gp.quicksum(
        beta_p * z_minus[1][j] for j in J_0_minus[1]) + gamma >= beta_p * sum(1 for _ in J_0_minus[1]) + beta_p * sum(
        1 for _ in J_le_minus[1]))
    model.addConstr((gp.quicksum(u[p] for p in range(dim)))-6 >= 0)
    # Set Objective function
    obj = gp.quicksum((z_plus[0][j] / N) for j in J_0_plus[0]) + sum(
        (1 / N) for _ in J_ge_plus[0]) - rho * gamma - lbd * (
                  gp.quicksum((w[p] - w_bar[p]) * (w[p] - w_bar[p]) for p in range(dim)) + (b - b_bar) * (
                  b - b_bar)) - lbd_2 * ((gp.quicksum(u[p] for p in range(dim)))-6)
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)
    if full_mip:
        model.setParam("Timelimit", 3600)
    else:
        if num_integer_vars <= 400:
            model.setParam("Timelimit", 300)
        elif num_integer_vars <= 800:
            model.setParam("Timelimit", 600)
        elif num_integer_vars <= 1200:
            model.setParam("Timelimit", 900)
        elif num_integer_vars <= 1600:
            model.setParam("Timelimit", 1800)
        else:
            model.setParam("Timelimit", 3600)
    model.setParam('LogFile',
                   dirname + '/LogFile' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(
                       iterations) + '.txt')
    model.optimize()
    model.write(
        dirname + '/Model' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(iterations) + ".lp")
    optimal_value = model.objVal
    optimality_gap = model.MIPGap
    optimal_w = [w[p].X for p in range(dim)]
    optimal_u = [u[p].X for p in range(dim)]
    optimal_b = b.X
    for i in range(obj_cons_num):
        optimal_z_plus[i] = {}
        optimal_z_minus[i] = {}
        for j in J_0_plus[i]:
            optimal_z_plus[i][j] = z_plus[i][j].X
        for j in J_ge_plus[i]:
            optimal_z_plus[i][j] = 1
        for j in J_le_plus[i]:
            optimal_z_plus[i][j] = 0

        for j in J_0_minus[i]:
            optimal_z_minus[i][j] = z_minus[i][j].X
        for j in J_ge_minus[i]:
            optimal_z_minus[i][j] = 1
        for j in J_le_minus[i]:
            optimal_z_minus[i][j] = 0
    with (open(dirname + '/Solution' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(
            iterations) + '.txt',
               'a') as f):
        violations = 0
        for i in range(obj_cons_num):
            for j in range(item_plus[i]):
                print("y:", y[j], file=f)
                if j in J_0_plus[i]:
                    print(str(j) + ' in ' + 'J_0_plus[' + str(i) + ']', file=f)
                elif j in J_ge_plus[i]:
                    print(str(j) + ' in ' + 'J_ge_plus[' + str(i) + ']', file=f)
                else:
                    print(str(j) + ' in ' + 'J_le_plus[' + str(i) + ']', file=f)
                print("z^+_" + str(i) + '_' + str(j) + '=', optimal_z_plus[i][j], file=f)
                if i == 0:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=', y[j] * (np.dot(optimal_w, X[j]) + optimal_b),
                          file=f)
                if i == 1:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=', min(y[j], np.dot(optimal_w, X[j]) + optimal_b),
                          file=f)
            for j in range(item_minus[i]):
                print("y:", y[j], file=f)
                if j in J_0_minus[i]:
                    print(str(j) + ' in ' + 'J_0_minus[' + str(i) + ']', file=f)
                elif j in J_ge_minus[i]:
                    print(str(j) + ' in ' + 'J_ge_minus[' + str(i) + ']', file=f)
                else:
                    print(str(j) + ' in ' + 'J_le_minus[' + str(i) + ']', file=f)
                print("z^-_" + str(i) + '_' + str(j) + '=', optimal_z_minus[i][j], file=f)
                if i == 1:
                    print('-\phi^-_' + str(i) + '_' + str(j) + '-epsilon =',
                          -(np.dot(optimal_w, X[j]) + optimal_b) - epsilon,
                          file=f)
                    if (np.dot(optimal_w, X[j]) + optimal_b < 0) & (np.dot(optimal_w, X[j]) + optimal_b > -epsilon):
                        violations += 1
                        print('\phi^-_' + str(i) + '_' + str(j) + '=', (np.dot(optimal_w, X[j]) + optimal_b),
                              'violates the assumption!',
                              file=f)
    real_TP, real_FP, real_TN, real_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(optimal_w, X[s]) + optimal_b >= 0) & (y[s] == 1):
            real_TP += 1
        elif (np.dot(optimal_w, X[s]) + optimal_b >= 0) & (y[s] == -1):
            real_FP += 1
        elif (np.dot(optimal_w, X[s]) + optimal_b < 0) & (y[s] == 1):
            real_FN += 1
        else:
            real_TN += 1
    real_results = {'recall': real_TP / (real_TP + real_FN) if (real_TP + real_FN) > 0 else 0.0,
                    'precision': real_TP / (real_TP + real_FP) if (real_TP + real_FP) > 0 else 0.0,
                    'accuracy': (real_TP + real_TN) / N}
    buffered_TP, buffered_FP, buffered_TN, buffered_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(optimal_w, X[s]) + optimal_b >= -1e-5) & (y[s] == 1):
            buffered_TP += 1
        elif (np.dot(optimal_w, X[s]) + optimal_b >= -1e-5) & (y[s] == -1):
            buffered_FP += 1
        elif (np.dot(optimal_w, X[s]) + optimal_b < -1e-5) & (y[s] == 1):
            buffered_FN += 1
        else:
            buffered_TN += 1
    buffered_results = {'recall': buffered_TP / (buffered_TP + buffered_FN) if (buffered_TP + buffered_FN) > 0 else 0.0,
                        'precision': buffered_TP / (buffered_TP + buffered_FP) if (buffered_TP + buffered_FP) > 0 else 0.0,
                        'accuracy': (buffered_TP + buffered_TN) / N}
    precision_in_constraint = (sum(optimal_z_plus[1][j] for j in J_0_plus[1]) + sum(1 for _ in J_ge_plus[1])) / (
            sum((1 - optimal_z_minus[1][j]) for j in J_0_minus[1]) + sum(1 for _ in J_le_minus[1]))
    counts_results = {
        'real_TP': real_TP, 'real_FP': real_FP, 'real_TN': real_TN, 'real_FN': real_FN,
        'buffered_TP': buffered_TP, 'buffered_FP': buffered_FP, 'buffered_TN': buffered_TN, 'buffered_FN': buffered_FN,
        'precision_in_constraint': precision_in_constraint, 'violations': violations
    }
    objective_function_terms = {
        'accuracy_in_obj': sum((optimal_z_plus[0][j] / N) for j in J_0_plus[0]) + sum((1 / N) for _ in J_ge_plus[0]),
        'gamma_in_obj': gamma.X,
        'regularization': lbd * (sum((optimal_w[p] - w_bar[p]) * (optimal_w[p] - w_bar[p]) for p in range(dim)) + (
                optimal_b - b_bar) * (optimal_b - b_bar)),
        'regularization_2': lbd_2 * ((sum(optimal_u[p] for p in range(dim)))-6)}
    return optimal_value, optimality_gap, optimal_w, optimal_b, optimal_z_plus, optimal_z_minus, objective_function_terms, real_results, buffered_results, counts_results
