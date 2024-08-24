import initial_sol
import initial_feasible_sol
import PIP_fixed
import PIP_iterations
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from datetime import datetime
import csv
import time
import pandas as pd


#
def epsilon_shrinkage(model, obj_cons_num, X, y, X_test, y_test, w_start, b_start, epsilon, M, rho, beta_p, lbd, lbd_2,
                      max_inner_iteration, max_outer_iteration, gap, sigma, base_rate,
                      enlargement_rate, shrinkage_rate, pip_max_rate, outer_dirname, full_mip):
    """
    Solve Algorithm IV: An Iterative Algorithm for Solving Fractional Heaviside Composite Optimization
    :param model: gp.Model("BinaryClassifier") in initialization
    :param obj_cons_num: 1 + # of constraints, 'I' in the paper
    :param X: X_train in training set
    :param y: y_train in training set
    :param X_test: X_test in test set
    :param y_test: y_test in test set
    :param w_start: (warm start) for setting: w[p].setAttr(gp.GRB.Attr.Start, w_start[p])
    :param b_start: (warm start) for setting: b.setAttr(gp.GRB.Attr.Start, b_start)
    :param epsilon: epsilon for outer iteration 0
    :param M: big M in constraints related to z
    :param rho: penalty coefficient for \gamma
    :param beta_p: lowerbound of precision
    :param lbd: coefficient for l-2 norm
    :param max_inner_iteration: max inner iteration
    :param max_outer_iteration: max outer iteration
    :param gap: the criterion of objective value remains unchanged in stopping rule
    :param sigma: shrinkage rate of epsilon
    :param base_rate: the ratio of integers in inner iteration 0 of PIP method
    :param enlargement_rate: enlargement rate of the ratio of integers in PIP
    :param shrinkage_rate: shrinkage rate of the ratio of integers in PIP
    :param pip_max_rate: max ratio of integers in PIP of stopping rule
    :param outer_dirname: output directory name
    :param full_mip: Bool, Full MIP or PIP
    :return: solution of the last outer iteration
    """
    fixed = False
    N = X.shape[0]
    item_plus = [N, N]
    item_minus = [0, N]
    objective_value_list = []
    optimality_gap_list = []
    w_list = []
    b_list = []
    objective_function_terms_list = []
    real_train_recall, real_train_precision, real_train_accuracy, buffered_train_recall, buffered_train_precision, buffered_train_accuracy, real_test_recall, real_test_precision, real_test_accuracy, buffered_test_recall, buffered_test_precision, buffered_test_accuracy = [], [], [], [], [], [], [], [], [], [], [], []
    epsilon = epsilon
    epsilon_nu = []
    objective_value_old = -2 * M
    objective_value = -M
    z_plus_start = None
    z_minus_start = None
    counts_results_list = []
    real_test_precision_violation_list = []
    buffered_test_precision_violation_list = []
    start_time = time.time()
    execution_time = []
    outer_iter_unchanged = 0
    for outer_iteration in range(max_outer_iteration):
        if abs(objective_value - objective_value_old) / abs(objective_value_old) <= gap:
            outer_iter_unchanged += 1
        epsilon = sigma * epsilon
        epsilon_nu.append(epsilon)
        objective_value_old = objective_value
        # current_datetime
        current_datetime = datetime.now()
        # dir_name
        dirname = outer_dirname + '/' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + '_outer_iter=' + str(
            outer_iteration)
        # make directory
        os.makedirs(dirname)
        os.makedirs(dirname + '/LogFile')
        os.makedirs(dirname + '/Model')
        os.makedirs(dirname + '/Solution')
        gamma_0, z_plus_start, z_minus_start = initial_feasible_sol.calculate_gamma(obj_cons_num,
                                                                                    X,
                                                                                    y,
                                                                                    w_start,
                                                                                    b_start,
                                                                                    beta_p,
                                                                                    epsilon)
        delta_1, delta_2 = initial_feasible_sol.calculate_delta(obj_cons_num=obj_cons_num, item_plus=item_plus,
                                                                item_minus=item_minus, X=X, y=y, weight=w_start,
                                                                bias=b_start, epsilon=epsilon, base_rate=base_rate)
        objective_value, optimality_gap, w_start, b_start, z_plus_start, z_minus_start, objective_function_terms, real_results, buffered_results, counts_results = PIP_iterations.pip_iterations(
            model,
            obj_cons_num,
            X,
            y,
            X_test,
            y_test,
            w_start,
            b_start,
            z_plus_start,
            z_minus_start,
            epsilon,
            delta_1,
            delta_2,
            gamma_0,
            M,
            rho,
            beta_p,
            lbd,
            lbd_2,
            max_inner_iteration,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            objective_value,
            outer_iteration,
            dirname,
            full_mip,
            fixed)
        end_time = time.time()
        execution_time.append(end_time - start_time)
        objective_value_list.append(objective_value)
        optimality_gap_list.append(optimality_gap)
        w_list.append(w_start)
        b_list.append(b_start)
        objective_function_terms_list.append(objective_function_terms)
        counts_results_list.append(counts_results)
        real_train_recall.append(real_results['recall'])
        real_train_precision.append(real_results['precision'])
        real_train_accuracy.append(real_results['accuracy'])
        buffered_train_recall.append(buffered_results['recall'])
        buffered_train_precision.append(buffered_results['precision'])
        buffered_train_accuracy.append(buffered_results['accuracy'])
        real_test_results, buffered_test_results = initial_sol.evaluate_classification(X_test, y_test, w_start, b_start)
        real_test_recall.append(real_test_results['recall'])
        real_test_precision.append(real_test_results['precision'])
        real_test_precision_violation = max(0, (beta_p - real_test_results['precision']) / beta_p)
        real_test_precision_violation_list.append(real_test_precision_violation)
        real_test_accuracy.append(real_test_results['accuracy'])
        buffered_test_recall.append(buffered_test_results['recall'])
        buffered_test_precision.append(buffered_test_results['precision'])
        buffered_test_precision_violation = max(0, (beta_p - buffered_test_results['precision']) / beta_p)
        buffered_test_precision_violation_list.append(buffered_test_precision_violation)
        buffered_test_accuracy.append(buffered_test_results['accuracy'])

        if outer_iter_unchanged >= 10:
            max_outer_iteration = outer_iteration + 1
            break

    iterative_time = [execution_time[0]]
    iterative_time += [execution_time[i] - execution_time[i - 1] for i in range(1, len(execution_time))]
    with open(outer_dirname + '/outer_results_beta_p='+str(beta_p)+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['outer_iteration', 'objective_value', 'optimality_gap', 'epsilon_nu', 'cumulative_time', 'iterative_time',
             'w', 'b',
             'accuracy_in_obj', 'gamma_in_obj', 'regularization', 'regularization_2',
             'real_TP', 'real_FP', 'real_TN', 'real_FN',
             'buffered_TP', 'buffered_FP', 'buffered_TN', 'buffered_FN',
             'precision_in_constraint', 'violations',
             'real_train_accuracy', 'real_train_precision', 'real_train_recall',
             'buffered_train_accuracy', 'buffered_train_precision', 'buffered_train_recall',
             'real_test_accuracy', 'real_test_precision', 'real_test_recall', 'real_test_precision_violation',
             'buffered_test_accuracy', 'buffered_test_precision', 'buffered_test_recall',
             'buffered_test_precision_violation'])
        for iteration in range(max_outer_iteration):
            writer.writerow(
                [iteration, objective_value_list[iteration], optimality_gap_list[iteration], epsilon_nu[iteration],
                 execution_time[iteration], iterative_time[iteration],
                 w_list[iteration], b_list[iteration],
                 objective_function_terms_list[iteration]['accuracy_in_obj'],
                 objective_function_terms_list[iteration]['gamma_in_obj'],
                 objective_function_terms_list[iteration]['regularization'],
                 objective_function_terms_list[iteration]['regularization_2'],
                 counts_results_list[iteration]['real_TP'], counts_results_list[iteration]['real_FP'],
                 counts_results_list[iteration]['real_TN'], counts_results_list[iteration]['real_FN'],
                 counts_results_list[iteration]['buffered_TP'], counts_results_list[iteration]['buffered_FP'],
                 counts_results_list[iteration]['buffered_TN'], counts_results_list[iteration]['buffered_FN'],
                 counts_results_list[iteration]['precision_in_constraint'],
                 counts_results_list[iteration]['violations'],
                 real_train_accuracy[iteration], real_train_precision[iteration], real_train_recall[iteration],
                 buffered_train_accuracy[iteration], buffered_train_precision[iteration],
                 buffered_train_recall[iteration],
                 real_test_accuracy[iteration], real_test_precision[iteration], real_test_recall[iteration],
                 real_test_precision_violation_list[iteration],
                 buffered_test_accuracy[iteration], buffered_test_precision[iteration],
                 buffered_test_recall[iteration], buffered_test_precision_violation_list[iteration]])
    with open(outer_dirname + '/outer_results_comparison.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['final_train', real_train_accuracy[max_outer_iteration - 1], real_train_precision[max_outer_iteration - 1],
             real_train_recall[max_outer_iteration - 1], buffered_train_accuracy[max_outer_iteration - 1],
             buffered_train_precision[max_outer_iteration - 1],
             buffered_train_recall[max_outer_iteration - 1]])
    y_binary = ((y + 1) / 2)
    y_test_binary = ((y_test + 1) / 2)
    initial_sol.precision_accuracy_curve(X, y_binary, w_start, b_start, num_thresholds=100,
                                         save_path=outer_dirname + '/outer_precision_accuracy_curve_final_train_beta_p=' + str(
                                             beta_p) + '.png')
    initial_sol.precision_accuracy_curve(X_test, y_test_binary, w_start, b_start, num_thresholds=100,
                                         save_path=outer_dirname + '/outer_precision_accuracy_curve_final_test_beta_p=' + str(
                                             beta_p) + '.png')
    return objective_value, w_start, b_start, z_plus_start, z_minus_start, counts_results_list[max_outer_iteration - 1][
        'precision_in_constraint']


def fixed_epsilon(model, obj_cons_num, X, y, X_test, y_test, w_start, b_start, epsilon, M, rho, beta_p, lbd, lbd_2,
                  max_inner_iteration, max_fixed_iteration, base_rate,
                  enlargement_rate, shrinkage_rate, pip_max_rate, fixed_dirname, full_mip):
    """
    Solve Fractional Heaviside Composite Optimization Problem with some fixed epsilons (no warm start)
    :param model:
    :param obj_cons_num:
    :param X:
    :param y:
    :param X_test:
    :param y_test:
    :param w_start:
    :param b_start:
    :param epsilon:
    :param M:
    :param rho:
    :param beta_p:
    :param lbd:
    :param max_inner_iteration:
    :param max_fixed_iteration:
    :param base_rate:
    :param enlargement_rate:
    :param shrinkage_rate:
    :param pip_max_rate:
    :param fixed_dirname:
    :param full_mip:
    :return:
    """
    fixed = True
    N = X.shape[0]
    item_plus = [N, N]
    item_minus = [0, N]
    objective_value_list = []
    optimality_gap_list = []
    w_list = []
    b_list = []
    objective_function_terms_list = []
    real_train_recall, real_train_precision, real_train_accuracy, buffered_train_recall, buffered_train_precision, buffered_train_accuracy, real_test_recall, real_test_precision, real_test_accuracy, buffered_test_recall, buffered_test_precision, buffered_test_accuracy = [], [], [], [], [], [], [], [], [], [], [], []
    epsilon = epsilon
    epsilon_list = []
    objective_value = -M
    z_plus_start = None
    z_minus_start = None
    counts_results_list = []
    real_test_precision_violation_list = []
    buffered_test_precision_violation_list = []
    start_time = time.time()
    execution_time = []
    for fixed_iteration in range(max_fixed_iteration):
        epsilon_q = epsilon * (10 ** (-fixed_iteration - 1))
        epsilon_list.append(epsilon_q)
        # current_datetime
        current_datetime = datetime.now()
        # dir_name
        dirname = fixed_dirname + '/' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + '_fixed_iter=' + str(
            fixed_iteration)
        # make directory
        os.makedirs(dirname)
        os.makedirs(dirname + '/LogFile')
        os.makedirs(dirname + '/Model')
        os.makedirs(dirname + '/Solution')
        gamma_0, z_plus_start, z_minus_start = initial_feasible_sol.calculate_gamma(obj_cons_num,
                                                                                    X,
                                                                                    y,
                                                                                    w_start,
                                                                                    b_start,
                                                                                    beta_p,
                                                                                    epsilon_q)
        delta_1, delta_2 = initial_feasible_sol.calculate_delta(obj_cons_num=obj_cons_num, item_plus=item_plus,
                                                                item_minus=item_minus, X=X, y=y, weight=w_start,
                                                                bias=b_start, epsilon=epsilon_q, base_rate=base_rate)
        objective_value, optimality_gap, w_current, b_current, z_plus_current, z_minus_current, objective_function_terms, real_results, buffered_results, counts_results = PIP_iterations.pip_iterations(
            model,
            obj_cons_num,
            X,
            y,
            X_test,
            y_test,
            w_start,
            b_start,
            z_plus_start,
            z_minus_start,
            epsilon_q,
            delta_1,
            delta_2,
            gamma_0,
            M,
            rho,
            beta_p,
            lbd,
            lbd_2,
            max_inner_iteration,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            objective_value,
            fixed_iteration,
            dirname,
            full_mip,
            fixed)
        end_time = time.time()
        execution_time.append(end_time - start_time)
        objective_value_list.append(objective_value)
        optimality_gap_list.append(optimality_gap)
        w_list.append(w_current)
        b_list.append(b_current)
        y_binary = ((y + 1) / 2)
        y_test_binary = ((y_test + 1) / 2)
        initial_sol.precision_accuracy_curve(X, y_binary, w_current, b_current, num_thresholds=100,
                                             save_path=dirname + '/fixed_iter=' + str(
                                                 fixed_iteration) + '_precision_accuracy_curve_train_beta_p=' + str(
                                                 beta_p) + '.png')
        initial_sol.precision_accuracy_curve(X_test, y_test_binary, w_current, b_current, num_thresholds=100,
                                             save_path=dirname + '/fixed_iter=' + str(
                                                 fixed_iteration) + '_precision_accuracy_curve_test_beta_p=' + str(
                                                 beta_p) + '.png')
        objective_function_terms_list.append(objective_function_terms)
        counts_results_list.append(counts_results)
        real_train_recall.append(real_results['recall'])
        real_train_precision.append(real_results['precision'])
        real_train_accuracy.append(real_results['accuracy'])
        buffered_train_recall.append(buffered_results['recall'])
        buffered_train_precision.append(buffered_results['precision'])
        buffered_train_accuracy.append(buffered_results['accuracy'])
        real_test_results, buffered_test_results = initial_sol.evaluate_classification(X_test, y_test, w_current, b_current)
        real_test_recall.append(real_test_results['recall'])
        real_test_precision.append(real_test_results['precision'])
        real_test_precision_violation = max(0, (beta_p - real_test_results['precision']) / beta_p)
        real_test_precision_violation_list.append(real_test_precision_violation)
        real_test_accuracy.append(real_test_results['accuracy'])
        buffered_test_recall.append(buffered_test_results['recall'])
        buffered_test_precision.append(buffered_test_results['precision'])
        buffered_test_precision_violation = max(0, (beta_p - buffered_test_results['precision']) / beta_p)
        buffered_test_precision_violation_list.append(buffered_test_precision_violation)
        buffered_test_accuracy.append(buffered_test_results['accuracy'])

    iterative_time = [execution_time[0]]
    iterative_time += [execution_time[i] - execution_time[i - 1] for i in range(1, len(execution_time))]
    with open(fixed_dirname + '/fixed_results_beta_p='+str(beta_p)+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['fixed_iteration', 'objective_value', 'optimality_gap', 'epsilon_q', 'cumulative_time', 'iterative_time',
             'w', 'b',
             'accuracy_in_obj', 'gamma_in_obj', 'regularization', 'regularization_2',
             'real_TP', 'real_FP', 'real_TN', 'real_FN',
             'buffered_TP', 'buffered_FP', 'buffered_TN', 'buffered_FN',
             'precision_in_constraint', 'violations',
             'real_train_accuracy', 'real_train_precision', 'real_train_recall',
             'buffered_train_accuracy', 'buffered_train_precision', 'buffered_train_recall',
             'real_test_accuracy', 'real_test_precision', 'real_test_recall', 'real_test_precision_violation',
             'buffered_test_accuracy', 'buffered_test_precision', 'buffered_test_recall',
             'buffered_test_precision_violation'])
        for iteration in range(max_fixed_iteration):
            writer.writerow(
                [iteration, objective_value_list[iteration], optimality_gap_list[iteration], epsilon_list[iteration],
                 execution_time[iteration], iterative_time[iteration],
                 w_list[iteration], b_list[iteration],
                 objective_function_terms_list[iteration]['accuracy_in_obj'],
                 objective_function_terms_list[iteration]['gamma_in_obj'],
                 objective_function_terms_list[iteration]['regularization'],
                 objective_function_terms_list[iteration]['regularization_2'],
                 counts_results_list[iteration]['real_TP'], counts_results_list[iteration]['real_FP'],
                 counts_results_list[iteration]['real_TN'], counts_results_list[iteration]['real_FN'],
                 counts_results_list[iteration]['buffered_TP'], counts_results_list[iteration]['buffered_FP'],
                 counts_results_list[iteration]['buffered_TN'], counts_results_list[iteration]['buffered_FN'],
                 counts_results_list[iteration]['precision_in_constraint'],
                 counts_results_list[iteration]['violations'],
                 real_train_accuracy[iteration], real_train_precision[iteration], real_train_recall[iteration],
                 buffered_train_accuracy[iteration], buffered_train_precision[iteration],
                 buffered_train_recall[iteration],
                 real_test_accuracy[iteration], real_test_precision[iteration], real_test_recall[iteration],
                 real_test_precision_violation_list[iteration],
                 buffered_test_accuracy[iteration], buffered_test_precision[iteration],
                 buffered_test_recall[iteration], buffered_test_precision_violation_list[iteration]])
    with open(fixed_dirname + '/fixed_results_comparison.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['final_train', real_train_accuracy[max_fixed_iteration - 1], real_train_precision[max_fixed_iteration - 1],
             real_train_recall[max_fixed_iteration - 1],
             buffered_train_accuracy[max_fixed_iteration - 1], buffered_train_precision[max_fixed_iteration - 1],
             buffered_train_recall[max_fixed_iteration - 1]])
    return objective_value, w_start, b_start, z_plus_start, z_minus_start, counts_results_list[max_fixed_iteration - 1][
        'precision_in_constraint']


def initialization(positive_size=1.0, negative_size=0.125, fixed_beta_p=0.1, outer_beta_p=0.1, file_path=None,
                   full_mip=False):
    """
    Initialization of all parameters in function 'epsilon_shrinkage'
    :param positive_size: # of positive samples / # of positives in dataset
    :param negative_size: # of negative samples / # of negatives in dataset
    :param beta_p: lowerbound of precision
    :param file_path: file path
    :param full_mip: Bool, Full MIP or PIP
    :return: parameters in function 'epsilon_shrinkage'
    """
    # model
    model = gp.Model("BinaryClassifier")
    model.setParam('IntegralityFocus', 1)
    model.setParam('NumericFocus', 3)
    model.setParam('FeasibilityTol', 1e-09)
    # obj_cons_num
    obj_cons_num = 2
    # X,y
    file_path = file_path
    # file_path = None
    n_iters = 1000
    results = initial_sol.train_and_evaluate(file_path, n_iters, positive_size=positive_size,
                                             negative_size=negative_size)
    X_train = results['X_train']
    y_train = 2 * results['y_train'] - 1
    y_train = y_train.values
    X_test = results['X_test']
    y_test = 2 * results['y_test'] - 1
    y_test = y_test.values
    # w_start, b_start
    w_start = results['weights']
    b_start = results['bias']
    # beta_p
    fixed_beta_p = fixed_beta_p  # beta_p > results['train_precision']
    outer_beta_p = outer_beta_p
    # M, rho, lbd
    M = 1e2
    rho = 100
    lbd = 0.1
    lbd_2 = 0.1
    # max_inner_iteration
    max_inner_iteration = 10
    # max_outer_iteration
    max_outer_iteration = 5
    max_fixed_iteration = 3
    # base_rate, enlargement_rate, shrinkage_rate, pip_max_rate
    if not full_mip:
        base_rate = 15
        pip_max_rate = 60
    else:
        base_rate = 100
        pip_max_rate = 100
    # epsilon
    epsilon = 1
    enlargement_rate = 1.2
    shrinkage_rate = 0.8
    # gap
    gap = 0.001
    # sigma
    sigma = 0.1
    # outer_dirname
    current_datetime = datetime.now()
    if full_mip:
        dir_mip = '_full_mip'
    else:
        dir_mip = '_pip'
    outer_dirname = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + dir_mip + '_outer_positive_size=' + str(
        round(positive_size, 2)) + '_negative_size=' + str(round(negative_size, 2)) + '_beta_p=' + str(
        round(outer_beta_p, 2))
    os.makedirs(outer_dirname)
    outer_result_file = outer_dirname + '/outer_results_comparison.csv'
    fixed_dirname = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + dir_mip + '_fixed_positive_size=' + str(
        round(positive_size, 2)) + '_negative_size=' + str(round(negative_size, 2)) + '_beta_p=' + str(
        round(fixed_beta_p, 2))
    os.makedirs(fixed_dirname)
    fixed_result_file = fixed_dirname + '/fixed_results_comparison.csv'
    initial_real_train_results, initial_buffered_train_results = initial_sol.evaluate_classification(X_train, y_train,
                                                                                                     w_start,
                                                                                                     b_start)
    initial_real_test_results, initial_buffered_test_results = initial_sol.evaluate_classification(X_test, y_test,
                                                                                                   w_start, b_start)
    with open(outer_result_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['', 'real_accuracy', 'real_precision',
             'real_recall', 'buffered_accuracy', 'buffered_precision', 'buffered_recall', 'real_test_violation',
             'buffered_test_violation'])
        writer.writerow(
            ['initial_train', results['train_accuracy'], results['train_precision'], results['train_recall'],
             initial_buffered_train_results['accuracy'], initial_buffered_train_results['precision'],
             initial_buffered_train_results['recall']])
        writer.writerow(
            ['initial_test', results['test_accuracy'], results['test_precision'], results['test_recall'],
             initial_buffered_test_results['accuracy'], initial_buffered_test_results['precision'],
             initial_buffered_test_results['recall']])
    initial_sol.precision_accuracy_curve(X_train, results['y_train'], w_start, b_start, num_thresholds=100,
                                         save_path=outer_dirname + '/outer_precision_accuracy_curve_initial_train_beta_p=' + str(
                                             outer_beta_p) + '.png')
    initial_sol.precision_accuracy_curve(X_test, results['y_test'], w_start, b_start, num_thresholds=100,
                                         save_path=outer_dirname + '/outer_precision_accuracy_curve_initial_test_beta_p=' + str(
                                             outer_beta_p) + '.png')
    with open(fixed_result_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['', 'real_accuracy', 'real_precision',
             'real_recall', 'buffered_accuracy', 'buffered_precision', 'buffered_recall', 'real_test_violation',
             'buffered_test_violation'])
        writer.writerow(
            ['initial_train', results['train_accuracy'], results['train_precision'], results['train_recall'],
             initial_buffered_train_results['accuracy'], initial_buffered_train_results['precision'],
             initial_buffered_train_results['recall']])
        writer.writerow(
            ['initial_test', results['test_accuracy'], results['test_precision'], results['test_recall'],
             initial_buffered_test_results['accuracy'], initial_buffered_test_results['precision'],
             initial_buffered_test_results['recall']])
    initial_sol.precision_accuracy_curve(X_train, results['y_train'], w_start, b_start, num_thresholds=100,
                                         save_path=fixed_dirname + '/fixed_precision_accuracy_curve_initial_train_beta_p=' + str(
                                             fixed_beta_p) + '.png')
    initial_sol.precision_accuracy_curve(X_test, results['y_test'], w_start, b_start, num_thresholds=100,
                                         save_path=fixed_dirname + '/fixed_precision_accuracy_curve_initial_test_beta_p=' + str(
                                             fixed_beta_p) + '.png')
    return (
        model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, fixed_beta_p,
        outer_beta_p, lbd, lbd_2,
        max_inner_iteration, max_outer_iteration, max_fixed_iteration,
        gap, sigma, base_rate, enlargement_rate, shrinkage_rate, pip_max_rate, outer_dirname, fixed_dirname,
        outer_result_file, fixed_result_file, full_mip)


def AHC_optimization():
    adult_file_path = 'C:/Users/zhengke/Desktop/24_07/20240704/BinaryClassification/adult/adult_processed.csv'
    rice_file_path = 'rice'
    bank_file_path = 'C:/Users/zhengke/Desktop/24_07/20240704/BinaryClassification/bank+marketing/bank/bank-full.csv'
    churn_file_path = 'churn'
    paras = [
        [500 / 3810, 500 / 3810, 0.95, 0.95, rice_file_path, True],
        [1000 / 3810, 1000 / 3810, 0.95, 0.95, rice_file_path, True],
        [1500 / 3810, 1500 / 3810, 0.95, 0.95, rice_file_path, True],
        [500 / 3810, 500 / 3810, 0.96, 0.96, rice_file_path, True],
        [1000 / 3810, 1000 / 3810, 0.96, 0.96, rice_file_path, True],
        [1500 / 3810, 1500 / 3810, 0.96, 0.96, rice_file_path, True],
        [500 / 3810, 500 / 3810, 0.97, 0.97, rice_file_path, True],
        [1000 / 3810, 1000 / 3810, 0.97, 0.97, rice_file_path, True],
        [1500 / 3810, 1500 / 3810, 0.97, 0.97, rice_file_path, True],
        [500 / 3810, 500 / 3810, 0.98, 0.98, rice_file_path, True],
        [1000 / 3810, 1000 / 3810, 0.98, 0.98, rice_file_path, True],
        [1500 / 3810, 1500 / 3810, 0.98, 0.98, rice_file_path, True],
        [500 / 3810, 500 / 3810, 0.95, 0.95, rice_file_path, False],
        [1000 / 3810, 1000 / 3810, 0.95, 0.95, rice_file_path, False],
        [1500 / 3810, 1500 / 3810, 0.95, 0.95, rice_file_path, False],
        [500 / 3810, 500 / 3810, 0.96, 0.96, rice_file_path, False],
        [1000 / 3810, 1000 / 3810, 0.96, 0.96, rice_file_path, False],
        [1500 / 3810, 1500 / 3810, 0.96, 0.96, rice_file_path, False],
        [500 / 3810, 500 / 3810, 0.97, 0.97, rice_file_path, False],
        [1000 / 3810, 1000 / 3810, 0.97, 0.97, rice_file_path, False],
        [1500 / 3810, 1500 / 3810, 0.97, 0.97, rice_file_path, False],
        [500 / 3810, 500 / 3810, 0.98, 0.98, rice_file_path, False],
        [1000 / 3810, 1000 / 3810, 0.98, 0.98, rice_file_path, False],
        [1500 / 3810, 1500 / 3810, 0.98, 0.98, rice_file_path, False]
        # [2000 / 3810, 2000 / 3810, 0.946058091286307, 0.946058091286307, rice_file_path, False],
        # [2500 / 3810, 2500 / 3810, 0.927335640138408, 0.927335640138408, rice_file_path, False]
    ]
    for para in paras:
        (model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, fixed_beta_p,
         outer_beta_p, lbd, lbd_2,
         max_inner_iteration, max_outer_iteration, max_fixed_iteration, gap, sigma, base_rate, enlargement_rate, shrinkage_rate,
         pip_max_rate,
         outer_dirname, fixed_dirname,
         outer_result_file, fixed_result_file, full_mip) = initialization(positive_size=para[0], negative_size=para[1],
                                                                          outer_beta_p=min(para[2], 1),
                                                                          fixed_beta_p=min(para[3], 1),
                                                                          file_path=para[4],
                                                                          full_mip=para[5])

        outer_objective_value, outer_weight, outer_bias, outer_z_plus, outer_z_minus, outer_precision_in_constraint = epsilon_shrinkage(
            model, obj_cons_num,
            X_train, y_train,
            X_test, y_test,
            w_start,
            b_start, epsilon, M,
            rho, outer_beta_p,
            lbd, lbd_2,
            max_inner_iteration,
            max_outer_iteration,
            gap, sigma,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            outer_dirname,
            full_mip)
        fixed_objective_value, fixed_weight, fixed_bias, fixed_z_plus, fixed_z_minus, fixed_precision_in_constraint = fixed_epsilon(
            model, obj_cons_num,
            X_train, y_train,
            X_test, y_test,
            w_start,
            b_start, epsilon, M,
            rho, fixed_beta_p,
            lbd, lbd_2,
            max_inner_iteration,
            max_fixed_iteration,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            fixed_dirname,
            full_mip)
        with open(outer_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            real_test_results, buffered_test_results = initial_sol.evaluate_classification(X_test, y_test, outer_weight,
                                                                                           outer_bias)
            real_test_precision_violation = max(0, (outer_beta_p - real_test_results['precision']) / outer_beta_p)
            buffered_test_precision_violation = max(0,
                                                    (outer_beta_p - buffered_test_results['precision']) / outer_beta_p)
            writer.writerow(
                ['final_test', real_test_results['accuracy'], real_test_results['precision'],
                 real_test_results['recall'],
                 buffered_test_results['accuracy'], buffered_test_results['precision'],
                 buffered_test_results['recall'], real_test_precision_violation, buffered_test_precision_violation])
        with open(fixed_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            real_test_results, buffered_test_results = initial_sol.evaluate_classification(X_test, y_test, fixed_weight,
                                                                                           fixed_bias)
            real_test_precision_violation = max(0, (fixed_beta_p - real_test_results['precision']) / fixed_beta_p)
            buffered_test_precision_violation = max(0,
                                                    (fixed_beta_p - buffered_test_results['precision']) / fixed_beta_p)
            writer.writerow(
                ['final_test', real_test_results['accuracy'], real_test_results['precision'],
                 real_test_results['recall'],
                 buffered_test_results['accuracy'], buffered_test_results['precision'],
                 buffered_test_results['recall'], real_test_precision_violation, buffered_test_precision_violation])


AHC_optimization()
