import train_and_evaluate
import initial_feasible_sol
import PIP_iterations
import os
from datetime import datetime
import csv
import time


def fixed_epsilon(model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, beta_p, lbd,
                  max_inner_iteration, max_fixed_times, base_rate,
                  enlargement_rate, shrinkage_rate, pip_max_rate, fixed_dirname):
    """
    Solve Fractional Heaviside Composite Optimization Problem with some fixed epsilons (no warm start)
    :param model:
    :param obj_cons_num:
    :param X_train:
    :param y_train:
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
    :param max_fixed_times:
    :param base_rate:
    :param enlargement_rate:
    :param shrinkage_rate:
    :param pip_max_rate:
    :param fixed_dirname:
    :return:
    """
    fixed = True
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
    epsilon_list = []
    objective_value = -M

    w_current = None
    b_current = None
    z_plus_current = None
    z_minus_current = None

    counts_results_list = []
    real_test_precision_violation_list = []
    buffered_test_precision_violation_list = []
    start_time = time.time()
    cumulative_time = []

    for fixed_times in range(max_fixed_times):
        epsilon_q = epsilon * (10 ** (-fixed_times - 1))
        epsilon_list.append(epsilon_q)

        current_datetime = datetime.now()

        dirname = fixed_dirname + '/' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + '_fixed_times=' + str(
            fixed_times)

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
                                                                                    epsilon_q)

        delta_1, delta_2 = initial_feasible_sol.calculate_delta(obj_cons_num=obj_cons_num, item_plus=item_plus,
                                                                item_minus=item_minus, X=X_train, y=y_train,
                                                                weight=w_start,
                                                                bias=b_start, epsilon=epsilon_q, base_rate=base_rate)

        objective_value, optimality_gap, w_current, b_current, z_plus_current, z_minus_current, objective_function_terms, real_train_result, buffered_train_result, counts_result = PIP_iterations.pip_iterations(
            model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, z_plus_start, z_minus_start,
            epsilon_q,
            delta_1, delta_2, gamma_0, M, rho, beta_p, lbd, max_inner_iteration, base_rate, enlargement_rate,
            shrinkage_rate, pip_max_rate, objective_value, fixed_times, dirname, fixed)
        end_time = time.time()

        cumulative_time.append(end_time - start_time)
        objective_value_list.append(objective_value)
        optimality_gap_list.append(optimality_gap)
        w_list.append(w_current)
        b_list.append(b_current)

        objective_function_terms_list.append(objective_function_terms)
        counts_results_list.append(counts_result)
        real_train_results_list.append(real_train_result)
        buffered_train_results_list.append(buffered_train_result)

        real_test_result, buffered_test_result = train_and_evaluate.evaluate_classification(X_test, y_test, w_current,
                                                                                            b_current)

        real_test_results_list.append(real_test_result)

        real_test_precision_violation = max(0, (beta_p - real_test_result['precision']) / beta_p if beta_p > 0 else 0.0)
        real_test_precision_violation_list.append(real_test_precision_violation)

        buffered_test_results_list.append(buffered_test_result)

        buffered_test_precision_violation = max(0, (beta_p - buffered_test_result['precision']) / beta_p if beta_p > 0 else 0.0)
        buffered_test_precision_violation_list.append(buffered_test_precision_violation)

    iterative_time = [cumulative_time[0]]
    iterative_time += [cumulative_time[i] - cumulative_time[i - 1] for i in range(1, len(cumulative_time))]

    with open(fixed_dirname + '/fixed_results_beta_p=' + str(beta_p) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['fixed_times', 'objective_value', 'optimality_gap', 'epsilon_q', 'cumulative_time', 'iterative_time',
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

        for iteration in range(max_fixed_times):
            writer.writerow(
                [iteration, objective_value_list[iteration], optimality_gap_list[iteration], epsilon_list[iteration],
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

    with open('AHC_all_result.csv', mode='a', newline='') as all_result:
        writer = csv.writer(all_result)
        for iteration in range(max_fixed_times):
            writer.writerow(
                [objective_value_list[iteration], optimality_gap_list[iteration], iterative_time[iteration], w_list[iteration], b_list[iteration],
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

    with open(fixed_dirname + '/fixed_results_comparison.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['final_train', real_train_results_list[max_fixed_times - 1]['accuracy'],
             real_train_results_list[max_fixed_times - 1]['precision'],
             real_train_results_list[max_fixed_times - 1]['recall'],
             buffered_train_results_list[max_fixed_times - 1]['accuracy'],
             buffered_train_results_list[max_fixed_times - 1]['precision'],
             buffered_train_results_list[max_fixed_times - 1]['recall']])

    return objective_value, w_current, b_current, z_plus_current, z_minus_current, counts_results_list[max_fixed_times - 1][
        'precision_in_constraint']
