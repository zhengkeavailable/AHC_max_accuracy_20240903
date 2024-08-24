import PIP_single_iter
import csv
import time
import initial_feasible_sol
import train_and_evaluate


def pip_iterations(model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, z_plus_start, z_minus_start, epsilon,
                   delta_1, delta_2, gamma_0, M, rho, beta_p, lbd, max_iteration, base_rate, enlargement_rate,
                   shrinkage_rate, pip_max_rate, objective_value, outer_or_fixed_iteration, dirname, fixed):
    """
    :param model:
    :param obj_cons_num:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
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
    :param max_iteration:
    :param base_rate:
    :param enlargement_rate:
    :param shrinkage_rate:
    :param pip_max_rate:
    :param objective_value:
    :param outer_or_fixed_iteration:
    :param dirname:
    :param fixed:
    :return:
    """
    gamma = gamma_0
    N = X_train.shape[0]
    item_plus = [N, N]
    item_minus = [0, N]

    objective_value_list = []
    optimality_gap_list = []

    w_bar = w_start
    b_bar = b_start
    w_list = []
    b_list = []

    objective_function_terms_list = []
    shrinkage_list = []
    delta_1_list = [delta_1]
    delta_2_list = [delta_2]

    real_train_results, buffered_train_results = [], []
    real_test_results, buffered_test_results = [], []

    counts_results_list = []
    real_test_precision_violation_list = []
    buffered_test_precision_violation_list = []
    start_time = time.time()
    execution_time = []
    iter_unchanged = 0

    for iteration in range(max_iteration):
        objective_value_old = objective_value
        objective_value, optimality_gap, w_start, b_start, z_plus_start, z_minus_start, objective_function_terms, real_results, buffered_results, counts_results = PIP_single_iter.pip_single_iter(
            model=model, obj_cons_num=obj_cons_num, X_train=X_train, y_train=y_train, w_bar=w_bar, b_bar=b_bar,
            w_start=w_start, b_start=b_start, z_plus_start=z_plus_start, z_minus_start=z_minus_start, epsilon=epsilon,
            delta_1=delta_1, delta_2=delta_2, gamma_0=gamma, M=M, rho=rho, beta_p=beta_p, lbd=lbd, iterations=iteration,
            outer_or_fixed_iteration=outer_or_fixed_iteration, dirname=dirname, fixed=fixed)
        end_time = time.time()

        execution_time.append(end_time - start_time)
        gamma = objective_function_terms['gamma_in_obj']

        objective_value_list.append(objective_value)
        optimality_gap_list.append(optimality_gap)
        w_list.append(w_start)
        b_list.append(b_start)

        objective_function_terms_list.append(objective_function_terms)
        counts_results_list.append(counts_results)

        real_train_results.append(real_results)
        buffered_train_results.append(buffered_results)

        real_test_result, buffered_test_result = train_and_evaluate.evaluate_classification(X_test, y_test, w_start, b_start)
        real_test_results.append(real_test_result)
        buffered_train_results.append(buffered_test_result)

        real_test_precision_violation = max(0, (beta_p - real_test_result['precision']) / beta_p)
        real_test_precision_violation_list.append(real_test_precision_violation)

        buffered_test_precision_violation = max(0, (beta_p - buffered_test_result['precision']) / beta_p)
        buffered_test_precision_violation_list.append(buffered_test_precision_violation)

        if objective_value - objective_value_old <= 1e-5:
            shrinkage_list.append(0)
            iter_unchanged += 1
            if enlargement_rate * base_rate <= pip_max_rate:
                base_rate = min(100, enlargement_rate * base_rate)
            else:
                base_rate = pip_max_rate
        else:
            shrinkage_list.append(1)
            base_rate = shrinkage_rate * base_rate
        delta_1, delta_2 = initial_feasible_sol.calculate_delta(obj_cons_num=obj_cons_num, item_plus=item_plus,
                                                                item_minus=item_minus, X=X_train, y=y_train, weight=w_start,
                                                                bias=b_start, epsilon=epsilon, base_rate=base_rate)
        delta_1_list.append(delta_1)
        delta_2_list.append(delta_2)
        if iter_unchanged >= 3:
            max_iteration = iteration + 1
            break
    if not fixed:
        iter_name = '/outer_iter='
    else:
        iter_name = '/fixed_times='

    with open(dirname + iter_name + str(outer_or_fixed_iteration) + '_results_beta_p='+str(beta_p)+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Iterations', 'objective_value', 'optimality_gap', 'shrinkage', 'delta_1', 'delta_2_', 'time',
             'w', 'b',
             'accuracy_in_obj', 'gamma_in_obj', 'regularization',
             'real_TP', 'real_FP', 'real_TN', 'real_FN',
             'buffered_TP', 'buffered_FP', 'buffered_TN', 'buffered_FN',
             'precision_in_constraint', 'violations',
             'real_train_accuracy', 'real_train_precision', 'real_train_recall',
             'buffered_train_accuracy', 'buffered_train_precision', 'buffered_train_recall',
             'real_test_accuracy', 'real_test_precision', 'real_test_recall', 'real_test_precision_violation',
             'buffered_test_accuracy', 'buffered_test_precision', 'buffered_test_recall',
             'buffered_test_precision_violation'])

        for iteration in range(max_iteration):
            writer.writerow(
                [iteration, objective_value_list[iteration], optimality_gap_list[iteration], shrinkage_list[iteration],
                 delta_1_list[iteration],
                 delta_2_list[iteration],
                 execution_time[iteration], w_list[iteration], b_list[iteration],
                 objective_function_terms_list[iteration]['accuracy_in_obj'],
                 objective_function_terms_list[iteration]['gamma_in_obj'],
                 objective_function_terms_list[iteration]['regularization'],
                 counts_results_list[iteration]['real_TP'], counts_results_list[iteration]['real_FP'],
                 counts_results_list[iteration]['real_TN'], counts_results_list[iteration]['real_FN'],
                 counts_results_list[iteration]['buffered_TP'], counts_results_list[iteration]['buffered_FP'],
                 counts_results_list[iteration]['buffered_TN'], counts_results_list[iteration]['buffered_FN'],
                 counts_results_list[iteration]['precision_in_constraint'],
                 counts_results_list[iteration]['violations'],
                 real_train_results[iteration]['accuracy'], real_train_results[iteration]['precision'], real_train_results[iteration]['recall'],
                 buffered_train_results[iteration]['accuracy'], buffered_train_results[iteration]['precision'],
                 buffered_train_results[iteration]['recall'],
                 real_test_results[iteration]['accuracy'], real_test_results[iteration]['precision'], real_test_results[iteration]['recall'],
                 real_test_precision_violation_list[iteration],
                 buffered_test_results[iteration]['accuracy'], buffered_test_results[iteration]['precision'],
                 buffered_test_results[iteration]['recall'], buffered_test_precision_violation_list[iteration]])

    real_train_result = real_train_results[max_iteration - 1]
    buffered_train_result = buffered_train_results[max_iteration - 1]

    final_optimality_gap = optimality_gap_list[max_iteration - 1]
    final_objective_function_terms = objective_function_terms_list[max_iteration - 1]
    final_counts_result = counts_results_list[max_iteration - 1]

    return objective_value, final_optimality_gap, w_start, b_start, z_plus_start, z_minus_start, final_objective_function_terms, real_train_result, buffered_train_result, final_counts_result
