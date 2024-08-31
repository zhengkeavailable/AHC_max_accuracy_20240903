import initial_feasible_sol
import train_and_evaluate
import epsilon_shrinkage
import fixed_epsilon
import full_MIP
import initialization
import csv
import time


def AHC_optimization():
    rice_file_path = 'rice'
    file_name_csv = 'AHC_file_name.csv'
    with open(file_name_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'lambda_1'])
    all_result_csv = 'AHC_all_result.csv'
    with open('AHC_all_result.csv', mode='a', newline='') as all_result:
        writer = csv.writer(all_result)
        writer.writerow(['objective_value', 'optimality_gap', 'total_time', 'w', 'b',
                         'accuracy_in_obj', 'gamma_in_obj', 'regularization_(NOT_)in_obj',
                         'real_TP', 'real_FP', 'real_TN', 'real_FN',
                         'buffered_TP', 'buffered_FP', 'buffered_TN', 'buffered_FN',
                         'precision_in_constraint', 'violations',
                         'real_train_accuracy', 'real_train_precision', 'real_train_recall',
                         'buffered_train_accuracy', 'buffered_train_precision', 'buffered_train_recall',
                         'real_test_accuracy', 'real_test_precision', 'real_test_recall',
                         'real_test_precision_violation',
                         'buffered_test_accuracy', 'buffered_test_precision', 'buffered_test_recall',
                         'buffered_test_precision_violation'])

    paras = [
        [500 / 3810, 500 / 3810, 0, 0, 0, rice_file_path, 1],
        [1000 / 3810, 1000 / 3810, 0, 0, 0, rice_file_path, 1],
        [1500 / 3810, 1500 / 3810, 0, 0, 0, rice_file_path, 1],

        # [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 10],
        # [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path, 10],
        # [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 10],
        # [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 10],
        # [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path, 10],
        # [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 10],
        # [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 10],
        # [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path, 10],
        # [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 10],
        # [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 10],
        # [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path, 10],
        # [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 10],
        #
        # [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 1],
        # [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path, 1],
        # [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 1],
        # [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 1],
        # [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path, 1],
        # [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 1],
        # [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 1],
        # [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path, 1],
        # [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 1],
        # [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 1],
        # [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path, 1],
        # [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 1],
        #
        # [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.1],
        # [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.1],
        # [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.1],
        # [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.1],
        # [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.1],
        # [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.1],
        # [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.1],
        # [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.1],
        # [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.1],
        # [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.1],
        # [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.1],
        # [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.1],
        #
        # [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.01],
        # [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.01],
        # [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.01],
        # [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.01],
        # [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.01],
        # [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.01],
        # [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.01],
        # [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.01],
        # [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.01],
        # [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.01],
        # [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.01],
        # [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.01],
        #
        # [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.001],
        # [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.001],
        # [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path, 0.001],
        # [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.001],
        # [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.001],
        # [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path, 0.001],
        # [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.001],
        # [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.001],
        # [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path, 0.001],
        # [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.001],
        # [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.001],
        # [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path, 0.001],
    ]

    for para in paras:
        (model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho,
         outer_beta_p, fixed_beta_p, full_mip_beta_p,
         lbd, max_inner_iteration, max_outer_iteration, max_fixed_times,
         gap, sigma, base_rate, enlargement_rate, shrinkage_rate, pip_max_rate,
         outer_dirname, fixed_dirname, full_mip_dirname,
         outer_result_file, fixed_result_file, full_mip_result_file) = initialization.initialization(
            positive_size=para[0],
            negative_size=para[1],
            outer_beta_p=min(para[2], 1),
            fixed_beta_p=min(para[3], 1),
            full_mip_beta_p=min(para[4], 1),
            file_path=para[5],
            lbd=para[6])

        with open(file_name_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([full_mip_dirname])
            writer.writerow([fixed_dirname, para[6]])
            writer.writerow([fixed_dirname, para[6]])
            writer.writerow([fixed_dirname, para[6]])
            writer.writerow([outer_dirname, para[6]])

        # full_mip
        full_mip_epsilon = epsilon * (10 ** (-4))
        # w_start = [5.132503792285429, -1.8973700746849882, -2.1945844415888707, -0.9350356485952136, 0.27014794029218314, -1.8047069870786145, -0.07174101609338465]
        # b_start = -0.0376720306484597
        gamma_0, z_plus_start, z_minus_start = initial_feasible_sol.calculate_gamma(obj_cons_num=obj_cons_num,
                                                                                    X_train=X_train,
                                                                                    y_train=y_train,
                                                                                    weights=w_start,
                                                                                    bias=b_start,
                                                                                    beta_p=full_mip_beta_p,
                                                                                    epsilon=full_mip_epsilon)

        full_mip_start_time = time.time()
        full_mip_objective_value, full_mip_optimality_gap, full_mip_weights, full_mip_bias, full_mip_z_plus, full_mip_z_minus, full_mip_objective_function_term, full_mip_real_train_result, full_mip_buffered_train_result, full_mip_counts_result = full_MIP.full_mip(
            model=model, obj_cons_num=obj_cons_num, X_train=X_train, y_train=y_train, w_start=w_start, b_start=b_start,
            z_plus_start=z_plus_start, z_minus_start=z_minus_start, epsilon=full_mip_epsilon, gamma_0=gamma_0, M=M,
            rho=rho,
            beta_p=full_mip_beta_p, lbd=lbd, dirname=full_mip_dirname)
        full_mip_end_time = time.time()
        execution_time = full_mip_end_time - full_mip_start_time

        full_mip_real_test_result, full_mip_buffered_test_result = train_and_evaluate.evaluate_classification(X_test,
                                                                                                              y_test,
                                                                                                              full_mip_weights,
                                                                                                              full_mip_bias)
        full_mip_real_test_precision_violation = max(0, (full_mip_beta_p - full_mip_real_test_result[
            'precision']) / full_mip_beta_p if full_mip_beta_p > 0 else 0.0)
        full_mip_buffered_test_precision_violation = max(0, (full_mip_beta_p - full_mip_buffered_test_result[
                                                             'precision']) / full_mip_beta_p if full_mip_beta_p > 0 else 0.0)

        full_MIP.output_full_mip(full_mip_objective_value, full_mip_optimality_gap, full_mip_epsilon,
                                 execution_time,
                                 full_mip_weights, full_mip_bias,
                                 full_mip_objective_function_term,
                                 full_mip_counts_result,
                                 full_mip_real_train_result,
                                 full_mip_buffered_train_result,
                                 full_mip_real_test_result,
                                 full_mip_real_test_precision_violation,
                                 full_mip_buffered_test_result,
                                 full_mip_buffered_test_precision_violation, full_mip_dirname, full_mip_beta_p)

        with open('AHC_all_result.csv', mode='a', newline='') as all_result:
            writer = csv.writer(all_result)
            writer.writerow(
                [full_mip_objective_value, full_mip_optimality_gap, execution_time, full_mip_weights, full_mip_bias,
                 full_mip_objective_function_term['accuracy_in_obj'],
                 full_mip_objective_function_term['gamma_in_obj'],
                 full_mip_objective_function_term['regularization'],
                 full_mip_counts_result['real_TP'],
                 full_mip_counts_result['real_FP'],
                 full_mip_counts_result['real_TN'],
                 full_mip_counts_result['real_FN'],
                 full_mip_counts_result['buffered_TP'],
                 full_mip_counts_result['buffered_FP'],
                 full_mip_counts_result['buffered_TN'],
                 full_mip_counts_result['buffered_FN'],
                 full_mip_counts_result['precision_in_constraint'],
                 full_mip_counts_result['violations'],
                 full_mip_real_train_result['accuracy'],
                 full_mip_real_train_result['precision'],
                 full_mip_real_train_result['recall'],
                 full_mip_buffered_train_result['accuracy'],
                 full_mip_buffered_train_result['precision'],
                 full_mip_buffered_train_result['recall'],
                 full_mip_real_test_result['accuracy'],
                 full_mip_real_test_result['precision'],
                 full_mip_real_test_result['recall'],
                 full_mip_real_test_precision_violation,
                 full_mip_buffered_test_result['accuracy'],
                 full_mip_buffered_test_result['precision'],
                 full_mip_buffered_test_result['recall'],
                 full_mip_buffered_test_precision_violation])

        with open(full_mip_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['final_test', full_mip_real_test_result['accuracy'], full_mip_real_test_result['precision'],
                 full_mip_real_test_result['recall'],
                 full_mip_buffered_test_result['accuracy'], full_mip_buffered_test_result['precision'],
                 full_mip_buffered_test_result['recall'], full_mip_real_test_precision_violation,
                 full_mip_buffered_test_precision_violation])

        # fixed epsilon
        fixed_objective_value, fixed_weights, fixed_bias, fixed_z_plus, fixed_z_minus, fixed_precision_in_constraint = fixed_epsilon.fixed_epsilon(
            model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, fixed_beta_p,
            lbd, max_inner_iteration, max_fixed_times, base_rate, enlargement_rate, shrinkage_rate, pip_max_rate,
            fixed_dirname)

        with open(fixed_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            fixed_real_test_result, fixed_buffered_test_result = train_and_evaluate.evaluate_classification(X_test,
                                                                                                            y_test,
                                                                                                            fixed_weights,
                                                                                                            fixed_bias)
            fixed_real_test_precision_violation = max(0, (
                    fixed_beta_p - fixed_real_test_result['precision']) / fixed_beta_p if fixed_beta_p > 0 else 0.0)
            fixed_buffered_test_precision_violation = max(0,
                                                          (fixed_beta_p - fixed_buffered_test_result[
                                                              'precision']) / fixed_beta_p if fixed_beta_p > 0 else 0.0)
            writer.writerow(
                ['final_test', fixed_real_test_result['accuracy'], fixed_real_test_result['precision'],
                 fixed_real_test_result['recall'],
                 fixed_buffered_test_result['accuracy'], fixed_buffered_test_result['precision'],
                 fixed_buffered_test_result['recall'], fixed_real_test_precision_violation,
                 fixed_buffered_test_precision_violation])

        # epsilon-shrinkage
        outer_objective_value, outer_weights, outer_bias, outer_z_plus, outer_z_minus, outer_precision_in_constraint = epsilon_shrinkage.epsilon_shrinkage(
            model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, outer_beta_p,
            lbd,
            max_inner_iteration, max_outer_iteration, gap, sigma, base_rate, enlargement_rate, shrinkage_rate,
            pip_max_rate, outer_dirname)

        with open(outer_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            outer_real_test_result, outer_buffered_test_result = train_and_evaluate.evaluate_classification(X_test,
                                                                                                            y_test,
                                                                                                            outer_weights,
                                                                                                            outer_bias)
            outer_real_test_precision_violation = max(0, (
                    outer_beta_p - outer_real_test_result['precision']) / outer_beta_p if outer_beta_p > 0 else 0.0)
            outer_buffered_test_precision_violation = max(0,
                                                          (outer_beta_p - outer_buffered_test_result[
                                                              'precision']) / outer_beta_p if outer_beta_p > 0 else 0.0)
            writer.writerow(
                ['final_test', outer_real_test_result['accuracy'], outer_real_test_result['precision'],
                 outer_real_test_result['recall'],
                 outer_buffered_test_result['accuracy'], outer_buffered_test_result['precision'],
                 outer_buffered_test_result['recall'], outer_real_test_precision_violation,
                 outer_buffered_test_precision_violation])


AHC_optimization()
