import train_and_evaluate
import initial_feasible_sol
import precision_and_accuracy_curve
import PIP_iterations
import os
from datetime import datetime
import csv
import time


def epsilon_shrinkage(model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, beta_p, lbd,
                      max_inner_iteration, max_outer_iteration, gap, sigma, base_rate,
                      enlargement_rate, shrinkage_rate, pip_max_rate, outer_dirname):
    """
    Solve Algorithm IV: An Iterative Algorithm for Solving Fractional Heaviside Composite Optimization
    :param model: gp.Model("BinaryClassifier") in initialization
    :param obj_cons_num: 1 + # of constraints, 'I' in the paper
    :param X_train: X_train in training set
    :param y_train: y_train in training set
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
    :return: solution of the last outer iteration
    """
    fixed = False
    N = X_train.shape[0]
    item_plus = [N, N]
    item_minus = [0, N]
    objective_value_list = []
    optimality_gap_list = []

    w_list = []
    b_list = []

    objective_function_terms_list = []
    real_train_results_list, buffered_train_results_list = [], []
    real_test_results_list, buffered_test_results_list = [], []

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
    cumulative_time = []
    outer_iter_unchanged = 0

    for outer_iteration in range(max_outer_iteration):
        if abs(objective_value - objective_value_old) / abs(objective_value_old) <= gap:
            outer_iter_unchanged += 1
        epsilon = sigma * epsilon
        epsilon_nu.append(epsilon)
        objective_value_old = objective_value

        current_datetime = datetime.now()

        dirname = outer_dirname + '/' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + '_outer_iter=' + str(
            outer_iteration)

        os.makedirs(dirname)
        os.makedirs(dirname + '/LogFile')
        os.makedirs(dirname + '/Model')
        os.makedirs(dirname + '/Solution')

        gamma_0, z_plus_start, z_minus_start = initial_feasible_sol.calculate_gamma(obj_cons_num,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    w_start,
                                                                                    b_start,
                                                                                    beta_p,
                                                                                    epsilon)

        delta_1, delta_2 = initial_feasible_sol.calculate_delta(obj_cons_num=obj_cons_num, item_plus=item_plus,
                                                                item_minus=item_minus, X=X_train, y=y_train, weight=w_start,
                                                                bias=b_start, epsilon=epsilon, base_rate=base_rate)

        objective_value, optimality_gap, w_start, b_start, z_plus_start, z_minus_start, objective_function_terms, real_train_result, buffered_train_result, counts_result = PIP_iterations.pip_iterations(
            model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, z_plus_start, z_minus_start, epsilon, delta_1,
            delta_2, gamma_0, M, rho, beta_p, lbd, max_inner_iteration, base_rate, enlargement_rate, shrinkage_rate,
            pip_max_rate, objective_value, outer_iteration, dirname, fixed)
        end_time = time.time()

        cumulative_time.append(end_time - start_time)
        objective_value_list.append(objective_value)
        optimality_gap_list.append(optimality_gap)
        w_list.append(w_start)
        b_list.append(b_start)

        objective_function_terms_list.append(objective_function_terms)
        counts_results_list.append(counts_result)

        real_train_results_list.append(real_train_result)
        buffered_train_results_list.append(buffered_train_result)

        real_test_result, buffered_test_result = train_and_evaluate.evaluate_classification(X_test, y_test, w_start, b_start)

        real_test_results_list.append(real_test_result)

        real_test_precision_violation = max(0, (beta_p - real_test_result['precision']) / beta_p)
        real_test_precision_violation_list.append(real_test_precision_violation)

        buffered_test_results_list.append(buffered_test_result)

        buffered_test_precision_violation = max(0, (beta_p - buffered_test_result['precision']) / beta_p)
        buffered_test_precision_violation_list.append(buffered_test_precision_violation)

        if outer_iter_unchanged >= 10:
            max_outer_iteration = outer_iteration + 1
            break

    iterative_time = [cumulative_time[0]]
    iterative_time += [cumulative_time[i] - cumulative_time[i - 1] for i in range(1, len(cumulative_time))]

    with open(outer_dirname + '/outer_results_beta_p='+str(beta_p)+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['outer_iteration', 'objective_value', 'optimality_gap', 'epsilon_nu', 'cumulative_time', 'iterative_time',
             'w', 'b',
             'accuracy_in_obj', 'gamma_in_obj', 'regularization_in_obj',
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
                 cumulative_time[iteration], iterative_time[iteration],
                 w_list[iteration], b_list[iteration],
                 objective_function_terms_list[iteration]['accuracy_in_obj'],
                 objective_function_terms_list[iteration]['gamma_in_obj'],
                 objective_function_terms_list[iteration]['regularization'],
                 counts_results_list[iteration]['real_TP'], counts_results_list[iteration]['real_FP'],
                 counts_results_list[iteration]['real_TN'], counts_results_list[iteration]['real_FN'],
                 counts_results_list[iteration]['buffered_TP'], counts_results_list[iteration]['buffered_FP'],
                 counts_results_list[iteration]['buffered_TN'], counts_results_list[iteration]['buffered_FN'],
                 counts_results_list[iteration]['precision_in_constraint'],
                 counts_results_list[iteration]['violations'],
                 real_train_results_list[iteration]['accuracy'], real_train_results_list[iteration]['precision'],
                 real_train_results_list[iteration]['recall'],
                 buffered_train_results_list[iteration]['accuracy'],
                 buffered_train_results_list[iteration]['precision'],
                 buffered_train_results_list[iteration]['recall'],
                 real_test_results_list[iteration]['accuracy'], real_test_results_list[iteration]['precision'],
                 real_test_results_list[iteration]['recall'],
                 real_test_precision_violation_list[iteration],
                 buffered_test_results_list[iteration]['accuracy'], buffered_test_results_list[iteration]['precision'],
                 buffered_test_results_list[iteration]['recall'], buffered_test_precision_violation_list[iteration]])

    with open(outer_dirname + '/outer_results_comparison.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['final_train', real_train_results_list[max_outer_iteration - 1]['accuracy'],
             real_train_results_list[max_outer_iteration - 1]['precision'],
             real_train_results_list[max_outer_iteration - 1]['recall'],
             buffered_train_results_list[max_outer_iteration - 1]['accuracy'],
             buffered_train_results_list[max_outer_iteration - 1]['precision'],
             buffered_train_results_list[max_outer_iteration - 1]['recall']])
        
    with open('AHC_all_result.csv', mode='a', newline='') as all_result:
        writer = csv.writer(all_result)
        writer.writerow([objective_value_list[max_outer_iteration - 1], optimality_gap_list[max_outer_iteration - 1], cumulative_time[max_outer_iteration - 1]])

    return objective_value, w_start, b_start, z_plus_start, z_minus_start, counts_results_list[max_outer_iteration - 1][
        'precision_in_constraint']
