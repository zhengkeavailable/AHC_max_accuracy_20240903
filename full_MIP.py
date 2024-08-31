import gurobipy as gp
from gurobipy import GRB
import numpy as np
import csv
import os


def full_mip(model, obj_cons_num, X_train, y_train, w_start, b_start, z_plus_start, z_minus_start, epsilon, gamma_0,
             M, rho, beta_p, lbd, dirname):
    """
    :param lbd:
    :param model:
    :param obj_cons_num: 
    :param X_train: 
    :param y_train: 
    :param w_start: 
    :param b_start: 
    :param z_plus_start: 
    :param z_minus_start: 
    :param epsilon: 
    :param gamma_0: 
    :param M: 
    :param rho: 
    :param beta_p: 
    :param dirname: 
    :return: 
    """
    model = model.copy()
    N = X_train.shape[0]
    dim = X_train.shape[1]

    item_plus = [N, N]
    item_minus = [0, N]

    z_plus = {}
    optimal_z_plus = {}
    z_minus = {}
    optimal_z_minus = {}

    w = model.addVars(dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
    b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    abs_diff_w = model.addVars(dim, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_diff_w")
    abs_diff_b = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_diff_b")
    gamma = model.addVar(lb=0, ub=gamma_0, vtype=GRB.CONTINUOUS, name="gamma")

    for p in range(dim):
        w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
        model.addConstr(w[p] - w_start[p] <= abs_diff_w[p])
        model.addConstr(w_start[p] - w[p] <= abs_diff_w[p])
    b.setAttr(gp.GRB.Attr.Start, b_start)
    model.addConstr(b - b_start <= abs_diff_b)
    model.addConstr(b_start - b <= abs_diff_b)

    for i in range(obj_cons_num):
        z_plus[i] = {}
        z_minus[i] = {}
        for j in range(item_plus[i]):
            z_plus[i][j] = model.addVar(vtype=GRB.BINARY, name="z_plus_" + str(i) + "_" + str(j))
        for j in range(item_minus[i]):
            z_minus[i][j] = model.addVar(vtype=GRB.BINARY, name="z_minus_" + str(i) + "_" + str(j))

    for i in range(obj_cons_num):
        for j in range(item_plus[i]):
            if i == 0:
                z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                model.addConstr(
                    y_train[j] * (gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b) >= -M * (1 - z_plus[i][j]))
            if i == 1:
                z_plus[i][j].setAttr(gp.GRB.Attr.Start, z_plus_start[i][j])
                model.addConstr(-M * (1 - z_plus[i][j]) <= y_train[j])
                model.addConstr(gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) + b >= -M * (1 - z_plus[i][j]))

        for j in range(item_minus[i]):
            if i == 1:
                z_minus[i][j].setAttr(gp.GRB.Attr.Start, z_minus_start[i][j])
                model.addConstr(
                    -gp.quicksum(w[p] * X_train[j][p] for p in range(dim)) - b >= -M * (1 - z_minus[i][j]) + epsilon)

    model.addConstr(gp.quicksum(z_plus[1][j] for j in range(N)) - gp.quicksum(
        beta_p * (1 - z_minus[1][j]) for j in range(N)) + gamma >= 0)

    obj = gp.quicksum((z_plus[0][j] / N) for j in range(N)) - rho * gamma
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    num_integer_vars = sum(1 for v in model.getVars() if v.vType == gp.GRB.BINARY)

    model.setParam("Timelimit", 3600)
    os.makedirs(os.path.join(dirname, 'LogFile'), exist_ok=True)
    model.setParam('LogFile', os.path.join(dirname, 'LogFile', 'log_file_full_mip.txt'))
    # model.setParam('LogFile',
    #                dirname + '/LogFile/log_file_full_mip.txt')
    model.optimize()
    os.makedirs(os.path.join(dirname, 'Model'), exist_ok=True)
    model.write(
        dirname + '/Model/model_full_mip.lp')

    objective_value = model.objVal
    optimality_gap = model.MIPGap
    optimal_w = [w[p].X for p in range(dim)]
    optimal_b = b.X

    for i in range(obj_cons_num):
        optimal_z_plus[i] = {}
        optimal_z_minus[i] = {}
        for j in range(N):
            optimal_z_plus[i][j] = z_plus[i][j].X
        if i == 1:
            for j in range(N):
                optimal_z_minus[i][j] = z_minus[i][j].X

    os.makedirs(os.path.join(dirname, 'Solution'), exist_ok=True)
    with (open(dirname + '/Solution/' + 'solution_full_mip.txt',
               'a') as f):
        violations = 0

        for i in range(obj_cons_num):
            for j in range(item_plus[i]):
                print("y_" + str(j) + '=', y_train[j], file=f)
                print("z^+_" + str(i) + '_' + str(j) + '=', optimal_z_plus[i][j], file=f)
                if i == 0:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=',
                          y_train[j] * (np.dot(optimal_w, X_train[j]) + optimal_b),
                          file=f)
                if i == 1:
                    print('\phi^+_' + str(i) + '_' + str(j) + '=',
                          min(y_train[j], np.dot(optimal_w, X_train[j]) + optimal_b),
                          file=f)
                    if (min(y_train[j], np.dot(optimal_w, X_train[j]) + optimal_b) >= 0) and (
                            optimal_z_plus[i][j] == 0):
                        print('z_plus_wrong!', file=f)
                    if (min(y_train[j], np.dot(optimal_w, X_train[j]) + optimal_b) < 0) and (
                            optimal_z_plus[i][j] == 1):
                        print('z_plus_wrong!', file=f)

            for j in range(item_minus[i]):
                print("y_" + str(j) + '=', y_train[j], file=f)
                print("z^-_" + str(i) + '_' + str(j) + '=', optimal_z_minus[i][j], file=f)
                if i == 1:
                    print('-\phi^-_' + str(i) + '_' + str(j) + '-epsilon =',
                          -(np.dot(optimal_w, X_train[j]) + optimal_b) - epsilon,
                          file=f)
                    if (-(np.dot(optimal_w, X_train[j]) + optimal_b) - epsilon >= 0) and (optimal_z_minus[i][j] == 0):
                        print('z_minus_wrong!', file=f)
                    if (-(np.dot(optimal_w, X_train[j]) + optimal_b) - epsilon < 0) and (optimal_z_minus[i][j] == 1):
                        print('z_minus_wrong!', file=f)
                    if (np.dot(optimal_w, X_train[j]) + optimal_b < 0) & (
                            np.dot(optimal_w, X_train[j]) + optimal_b > -epsilon):
                        violations += 1
                        print('\phi^-_' + str(i) + '_' + str(j) + '=', (np.dot(optimal_w, X_train[j]) + optimal_b),
                              'violates the assumption!',
                              file=f)

    real_TP, real_FP, real_TN, real_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(optimal_w, X_train[s]) + optimal_b >= 0) & (y_train[s] == 1):
            real_TP += 1
        elif (np.dot(optimal_w, X_train[s]) + optimal_b >= 0) & (y_train[s] == -1):
            real_FP += 1
        elif (np.dot(optimal_w, X_train[s]) + optimal_b < 0) & (y_train[s] == 1):
            real_FN += 1
        else:
            real_TN += 1

    real_train_result = {'recall': real_TP / (real_TP + real_FN) if (real_TP + real_FN) > 0 else 0.0,
                         'precision': real_TP / (real_TP + real_FP) if (real_TP + real_FP) > 0 else 0.0,
                         'accuracy': (real_TP + real_TN) / N}

    buffered_TP, buffered_FP, buffered_TN, buffered_FN = 0, 0, 0, 0
    for s in range(N):
        if (np.dot(optimal_w, X_train[s]) + optimal_b >= -1e-5) & (y_train[s] == 1):
            buffered_TP += 1
        elif (np.dot(optimal_w, X_train[s]) + optimal_b >= -1e-5) & (y_train[s] == -1):
            buffered_FP += 1
        elif (np.dot(optimal_w, X_train[s]) + optimal_b < -1e-5) & (y_train[s] == 1):
            buffered_FN += 1
        else:
            buffered_TN += 1

    buffered_train_result = {
        'recall': buffered_TP / (buffered_TP + buffered_FN) if (buffered_TP + buffered_FN) > 0 else 0.0,
        'precision': buffered_TP / (buffered_TP + buffered_FP) if (buffered_TP + buffered_FP) > 0 else 0.0,
        'accuracy': (buffered_TP + buffered_TN) / N}

    precision_in_constraint = sum(optimal_z_plus[1][j] for j in range(N)) / (
        sum((1 - optimal_z_minus[1][j]) for j in range(N)))
    counts_results = {
        'real_TP': real_TP, 'real_FP': real_FP, 'real_TN': real_TN, 'real_FN': real_FN,
        'buffered_TP': buffered_TP, 'buffered_FP': buffered_FP, 'buffered_TN': buffered_TN, 'buffered_FN': buffered_FN,
        'precision_in_constraint': precision_in_constraint, 'violations': violations
    }

    objective_function_term = {
        'accuracy_in_obj': sum((optimal_z_plus[0][j] / N) for j in range(N)),
        'gamma_in_obj': gamma.X,
        'regularization': lbd * (gp.quicksum(abs_diff_w[p].X for p in range(dim)) + abs_diff_b.X)}

    return objective_value, optimality_gap, optimal_w, optimal_b, optimal_z_plus, optimal_z_minus, objective_function_term, real_train_result, buffered_train_result, counts_results


def output_full_mip(objective_value, optimality_gap, full_mip_epsilon,
                    execution_time,
                    w, b,
                    objective_function_term,
                    counts_result,
                    real_train_result,
                    buffered_train_result,
                    real_test_result,
                    real_test_precision_violation,
                    buffered_test_result,
                    buffered_test_precision_violation, full_mip_dirname, full_mip_beta_p):
    with open(full_mip_dirname + '/full_mip_results_beta_p=' + str(full_mip_beta_p) + '.csv', mode='w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['iteration', 'objective_value', 'optimality_gap', 'epsilon', 'cumulative_time', 'time',
             'w', 'b',
             'accuracy_in_obj', 'gamma_in_obj', 'regularization_NOT_in_obj',
             'real_TP', 'real_FP', 'real_TN', 'real_FN',
             'buffered_TP', 'buffered_FP', 'buffered_TN', 'buffered_FN',
             'precision_in_constraint', 'violations',
             'real_train_accuracy', 'real_train_precision', 'real_train_recall',
             'buffered_train_accuracy', 'buffered_train_precision', 'buffered_train_recall',
             'real_test_accuracy', 'real_test_precision', 'real_test_recall', 'real_test_precision_violation',
             'buffered_test_accuracy', 'buffered_test_precision', 'buffered_test_recall',
             'buffered_test_precision_violation'])
        writer.writerow(
            [0, objective_value, optimality_gap, full_mip_epsilon,
             execution_time, execution_time,
             w, b,
             objective_function_term['accuracy_in_obj'],
             objective_function_term['gamma_in_obj'],
             objective_function_term['regularization'],
             counts_result['real_TP'], counts_result['real_FP'],
             counts_result['real_TN'], counts_result['real_FN'],
             counts_result['buffered_TP'], counts_result['buffered_FP'],
             counts_result['buffered_TN'], counts_result['buffered_FN'],
             counts_result['precision_in_constraint'],
             counts_result['violations'],
             real_train_result['accuracy'], real_train_result['precision'],
             real_train_result['recall'],
             buffered_train_result['accuracy'],
             buffered_train_result['precision'],
             buffered_train_result['recall'],
             real_test_result['accuracy'], real_test_result['precision'],
             real_test_result['recall'],
             real_test_precision_violation,
             buffered_test_result['accuracy'], buffered_test_result['precision'],
             buffered_test_result['recall'], buffered_test_precision_violation])
