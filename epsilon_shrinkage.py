import train_and_evaluate
import initial_feasible_sol
import precision_and_accuracy_curve
import PIP_iterations
import os
from datetime import datetime
import csv
import time


def epsilon_shrinkage(model, obj_cons_num, X, y, X_test, y_test, w_start, b_start, epsilon, M, rho, beta_p, lbd,
                      max_inner_iteration, max_outer_iteration, gap, sigma, base_rate,
                      enlargement_rate, shrinkage_rate, pip_max_rate, outer_dirname):
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
            max_inner_iteration,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            objective_value,
            outer_iteration,
            dirname,
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
        real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test, w_start, b_start)
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
                 execution_time[iteration], iterative_time[iteration],
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
    return objective_value, w_start, b_start, z_plus_start, z_minus_start, counts_results_list[max_outer_iteration - 1][
        'precision_in_constraint']
