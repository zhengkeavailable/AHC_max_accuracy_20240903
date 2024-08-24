import train_and_evaluate
import initial_feasible_sol
import precision_and_accuracy_curve
import PIP_iterations
import os
from datetime import datetime
import csv
import time


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
        precision_and_accuracy_curve.precision_accuracy_curve(X, y_binary, w_current, b_current, num_thresholds=100,
                                             save_path=dirname + '/fixed_iter=' + str(
                                                 fixed_iteration) + '_precision_accuracy_curve_train_beta_p=' + str(
                                                 beta_p) + '.png')
        precision_and_accuracy_curve.precision_accuracy_curve(X_test, y_test_binary, w_current, b_current, num_thresholds=100,
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
        real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test, w_current, b_current)
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
