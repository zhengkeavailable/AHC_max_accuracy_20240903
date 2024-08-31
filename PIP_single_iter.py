import gurobipy as gp
from gurobipy import GRB
import numpy as np
import MIP_callback


def pip_single_iter(model, obj_cons_num, X_train, y_train, w_bar, b_bar, w_start, b_start, z_plus_start, z_minus_start,
                    epsilon,
                    delta_1,
                    delta_2, gamma_0,
                    M, rho, beta_p, lbd, iterations, outer_or_fixed_iteration, dirname, fixed):
    """
    :param model:
    :param obj_cons_num:
    :param X_train:
    :param y_train:
    :param w_bar:
    :param b_bar:
    :param w_start:
    :param b_start:
    :param z_plus_start:
    :param z_minus_start:
    :param epsilon:
    :param delta_1:
    :param delta_2:
    :param gamma_0:
    :param M:
    :param rho:
    :param beta_p:
    :param lbd:
    :param iterations:
    :param outer_or_fixed_iteration:
    :param dirname:
    :param fixed:
    :return:
    """
    if not fixed:
        iter_name = '/outer_iter='
    else:
        iter_name = '/fixed_times='

    model = model.copy()
    N = X_train.shape[0]
    dim = X_train.shape[1]

    item_plus = [N, N]
    item_minus = [0, N]

    z_plus = {}
    solution_z_plus = {}
    z_minus = {}
    solution_z_minus = {}

    J_0_plus, J_ge_plus, J_le_plus, J_0_minus, J_ge_minus, J_le_minus = {}, {}, {}, {}, {}, {}

    with open(dirname + '/Solution' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(
            iterations) + '.txt',
              'a') as f:
        print("delta_1,delta_2,epsilon:", delta_1, delta_2, epsilon, file=f)

    w = model.addVars(dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
    b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    abs_diff_w = model.addVars(dim, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_diff_w")
    abs_diff_b = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_diff_b")
    gamma = model.addVar(lb=0, ub=gamma_0, vtype=GRB.CONTINUOUS, name="gamma")

    for p in range(dim):
        w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
        model.addConstr(w[p] - w_bar[p] <= abs_diff_w[p])
        model.addConstr(w_bar[p] - w[p] <= abs_diff_w[p])
    b.setAttr(gp.GRB.Attr.Start, b_start)
    model.addConstr(b - b_bar <= abs_diff_b)
    model.addConstr(b_bar - b <= abs_diff_b)

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
                if y_train[j] * (np.dot(w_start, X_train[j]) + b_start) >= delta_1:
                    J_ge_plus[i].append(j)
                    model.remove(z_plus[i][j])
                    model.addConstr(y_train[j] * (gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b) >= 0)
                elif y_train[j] * (np.dot(w_start, X_train[j]) + b_start) <= -delta_2:
                    J_le_plus[i].append(j)
                    model.remove(z_plus[i][j])
                else:
                    J_0_plus[i].append(j)
                    z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                    model.addConstr(
                        y_train[j] * (gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b) >= -M * (
                                    1 - z_plus[i][j]))

            if i == 1:
                if min(y_train[j], np.dot(w_start, X_train[j]) + b_start) >= delta_1:
                    J_ge_plus[i].append(j)
                    model.remove(z_plus[i][j])
                    model.addConstr(gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b >= 0)
                elif min(y_train[j], np.dot(w_start, X_train[j]) + b_start) <= -delta_2:
                    J_le_plus[i].append(j)
                    model.remove(z_plus[i][j])
                else:
                    J_0_plus[i].append(j)
                    z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                    model.addConstr(-M * (1 - z_plus[i][j]) <= y_train[j])
                    model.addConstr(
                        gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b >= -M * (1 - z_plus[i][j]))

        for j in range(item_minus[i]):
            if i == 1:
                if -(np.dot(w_start, X_train[j]) + b_start) - epsilon >= delta_1:
                    J_ge_minus[i].append(j)
                    model.remove(z_minus[i][j])
                    model.addConstr(-gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) - b >= epsilon)
                elif -(np.dot(w_start, X_train[j]) + b_start) - epsilon <= -delta_2:
                    J_le_minus[i].append(j)
                    model.remove(z_minus[i][j])
                else:
                    J_0_minus[i].append(j)
                    z_minus[i][j].setAttr(gp.GRB.Attr.Start, z_minus_start[i][j])
                    model.addConstr(
                        -gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) - b >= -M * (
                                    1 - z_minus[i][j]) + epsilon)

    # model.addConstr(gp.quicksum(z_plus[1][j] for j in J_0_plus[1]) + sum(1 for _ in J_ge_plus[1]) + gp.quicksum(
    #     beta_p * z_minus[1][j] for j in J_0_minus[1]) + gamma >= beta_p * sum(1 for _ in J_0_minus[1]) + beta_p * sum(
    #     1 for _ in J_le_minus[1]))

    # Set Objective function
    if not fixed:
        obj = gp.quicksum((z_plus[0][j] / N) for j in J_0_plus[0]) + sum(
            (1 / N) for _ in J_ge_plus[0]) - rho * gamma - lbd * (
                      gp.quicksum(abs_diff_w[p] for p in range(dim)) + abs_diff_b)
    else:
        obj = gp.quicksum((z_plus[0][j] / N) for j in J_0_plus[0]) + sum(
            (1 / N) for _ in J_ge_plus[0]) - rho * gamma

    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

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
    # model.optimize(MIP_callback.mip_callback)
    model.optimize()
    model.write(
        dirname + '/Model' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(iterations) + ".lp")

    objective_value = model.objVal
    optimality_gap = model.MIPGap
    solution_w = [w[p].X for p in range(dim)]
    solution_b = b.X

    for i in range(obj_cons_num):
        solution_z_plus[i] = {}
        solution_z_minus[i] = {}

        for j in J_0_plus[i]:
            solution_z_plus[i][j] = z_plus[i][j].X
        for j in J_ge_plus[i]:
            solution_z_plus[i][j] = 1
        for j in J_le_plus[i]:
            solution_z_plus[i][j] = 0

        for j in J_0_minus[i]:
            solution_z_minus[i][j] = z_minus[i][j].X
        for j in J_ge_minus[i]:
            solution_z_minus[i][j] = 1
        for j in J_le_minus[i]:
            solution_z_minus[i][j] = 0

    with (open(dirname + '/Solution' + iter_name + str(outer_or_fixed_iteration) + '_inner_iter=' + str(
            iterations) + '.txt',
               'a') as f):
        violations = 0
        for i in range(obj_cons_num):
            for j in range(item_plus[i]):

                print("y_" + str(j) +'=', y_train[j], file=f)
                if j in J_0_plus[i]:
                    print(str(j) + ' in ' + 'J_0_plus[' + str(i) + ']', file=f)
                elif j in J_ge_plus[i]:
                    print(str(j) + ' in ' + 'J_ge_plus[' + str(i) + ']', file=f)
                else:
                    print(str(j) + ' in ' + 'J_le_plus[' + str(i) + ']', file=f)
                print("z^+_" + str(i) + '_' + str(j) + '=', solution_z_plus[i][j], file=f)

                if i == 0:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=',
                          y_train[j] * (np.dot(solution_w, X_train[j]) + solution_b),
                          file=f)
                if i == 1:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=',
                          min(y_train[j], np.dot(solution_w, X_train[j]) + solution_b),
                          file=f)
            for j in range(item_minus[i]):

                print("y_" + str(j) +'=', y_train[j], file=f)
                if j in J_0_minus[i]:
                    print(str(j) + ' in ' + 'J_0_minus[' + str(i) + ']', file=f)
                elif j in J_ge_minus[i]:
                    print(str(j) + ' in ' + 'J_ge_minus[' + str(i) + ']', file=f)
                else:
                    print(str(j) + ' in ' + 'J_le_minus[' + str(i) + ']', file=f)
                print("z^-_" + str(i) + '_' + str(j) + '=', solution_z_minus[i][j], file=f)

                if i == 1:
                    print('-\phi^-_' + str(i) + '_' + str(j) + '-epsilon =',
                          -(np.dot(solution_w, X_train[j]) + solution_b) - epsilon,
                          file=f)
                    if (np.dot(solution_w, X_train[j]) + solution_b < 0) & (
                            np.dot(solution_w, X_train[j]) + solution_b > -epsilon):
                        violations += 1
                        print('\phi^-_' + str(i) + '_' + str(j) + '=', (np.dot(solution_w, X_train[j]) + solution_b),
                              'violates the assumption!',
                              file=f)

    real_TP, real_FP, real_TN, real_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(solution_w, X_train[s]) + solution_b >= 0) & (y_train[s] == 1):
            real_TP += 1
        elif (np.dot(solution_w, X_train[s]) + solution_b >= 0) & (y_train[s] == -1):
            real_FP += 1
        elif (np.dot(solution_w, X_train[s]) + solution_b < 0) & (y_train[s] == 1):
            real_FN += 1
        else:
            real_TN += 1

    real_train_result = {'recall': real_TP / (real_TP + real_FN) if (real_TP + real_FN) > 0 else 0.0,
                         'precision': real_TP / (real_TP + real_FP) if (real_TP + real_FP) > 0 else 0.0,
                         'accuracy': (real_TP + real_TN) / N}

    buffered_TP, buffered_FP, buffered_TN, buffered_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(solution_w, X_train[s]) + solution_b >= -1e-5) & (y_train[s] == 1):
            buffered_TP += 1
        elif (np.dot(solution_w, X_train[s]) + solution_b >= -1e-5) & (y_train[s] == -1):
            buffered_FP += 1
        elif (np.dot(solution_w, X_train[s]) + solution_b < -1e-5) & (y_train[s] == 1):
            buffered_FN += 1
        else:
            buffered_TN += 1

    buffered_train_result = {
        'recall': buffered_TP / (buffered_TP + buffered_FN) if (buffered_TP + buffered_FN) > 0 else 0.0,
        'precision': buffered_TP / (buffered_TP + buffered_FP) if (buffered_TP + buffered_FP) > 0 else 0.0,
        'accuracy': (buffered_TP + buffered_TN) / N}

    precision_in_constraint = (sum(solution_z_plus[1][j] for j in J_0_plus[1]) + sum(1 for _ in J_ge_plus[1])) / (
            sum((1 - solution_z_minus[1][j]) for j in J_0_minus[1]) + sum(1 for _ in J_le_minus[1]))

    counts_results = {
        'real_TP': real_TP, 'real_FP': real_FP, 'real_TN': real_TN, 'real_FN': real_FN,
        'buffered_TP': buffered_TP, 'buffered_FP': buffered_FP, 'buffered_TN': buffered_TN, 'buffered_FN': buffered_FN,
        'precision_in_constraint': precision_in_constraint, 'violations': violations
    }

    # in fact, regularization is not included in fixed-epsilon problem, but we still calculate this term for final result comparison
    objective_function_term = {
        'accuracy_in_obj': sum((solution_z_plus[0][j] / N) for j in J_0_plus[0]) + sum((1 / N) for _ in J_ge_plus[0]),
        'gamma_in_obj': gamma.X,
        'regularization': lbd * (gp.quicksum(abs_diff_w[p].X for p in range(dim)) + abs_diff_b.X)}
    return objective_value, optimality_gap, solution_w, solution_b, solution_z_plus, solution_z_minus, objective_function_term, real_train_result, buffered_train_result, counts_results
