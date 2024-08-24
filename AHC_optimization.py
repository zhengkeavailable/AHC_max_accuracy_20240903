import initial_feasible_sol
import train_and_evaluate
import epsilon_shrinkage
import fixed_epsilon
import full_MIP
import initialization
import csv


def AHC_optimization():
    rice_file_path = 'rice'
    paras = [
        [500 / 3810, 500 / 3810, 0.95, 0.95, 0.95, rice_file_path],
        [1000 / 3810, 1000 / 3810, 0.95, 0.95, 0.95, rice_file_path],
        [1500 / 3810, 1500 / 3810, 0.95, 0.95, 0.95, rice_file_path],
        [500 / 3810, 500 / 3810, 0.96, 0.96, 0.96, rice_file_path],
        [1000 / 3810, 1000 / 3810, 0.96, 0.96, 0.96, rice_file_path],
        [1500 / 3810, 1500 / 3810, 0.96, 0.96, 0.96, rice_file_path],
        [500 / 3810, 500 / 3810, 0.97, 0.97, 0.97, rice_file_path],
        [1000 / 3810, 1000 / 3810, 0.97, 0.97, 0.97, rice_file_path],
        [1500 / 3810, 1500 / 3810, 0.97, 0.97, 0.97, rice_file_path],
        [500 / 3810, 500 / 3810, 0.98, 0.98, 0.98, rice_file_path],
        [1000 / 3810, 1000 / 3810, 0.98, 0.98, 0.98, rice_file_path],
        [1500 / 3810, 1500 / 3810, 0.98, 0.98, 0.98, rice_file_path]
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
            file_path=para[5])
        # outer
        outer_objective_value, outer_weight, outer_bias, outer_z_plus, outer_z_minus, outer_precision_in_constraint = epsilon_shrinkage.epsilon_shrinkage(
            model, obj_cons_num,
            X_train, y_train,
            X_test, y_test,
            w_start,
            b_start, epsilon, M,
            rho, outer_beta_p,
            lbd,
            max_inner_iteration,
            max_outer_iteration,
            gap, sigma,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            outer_dirname)
        with open(outer_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test,
                                                                                                  outer_weight,
                                                                                                  outer_bias)
            real_test_precision_violation = max(0, (outer_beta_p - real_test_results['precision']) / outer_beta_p)
            buffered_test_precision_violation = max(0,
                                                    (outer_beta_p - buffered_test_results[
                                                        'precision']) / outer_beta_p)
            writer.writerow(
                ['final_test', real_test_results['accuracy'], real_test_results['precision'],
                 real_test_results['recall'],
                 buffered_test_results['accuracy'], buffered_test_results['precision'],
                 buffered_test_results['recall'], real_test_precision_violation, buffered_test_precision_violation])
        # fixed
        fixed_objective_value, fixed_weight, fixed_bias, fixed_z_plus, fixed_z_minus, fixed_precision_in_constraint = fixed_epsilon.fixed_epsilon(
            model, obj_cons_num,
            X_train, y_train,
            X_test, y_test,
            w_start,
            b_start, epsilon, M,
            rho, fixed_beta_p,
            lbd,
            max_inner_iteration,
            max_fixed_times,
            base_rate,
            enlargement_rate,
            shrinkage_rate,
            pip_max_rate,
            fixed_dirname)
        with open(fixed_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test,
                                                                                                  fixed_weight,
                                                                                                  fixed_bias)
            real_test_precision_violation = max(0, (fixed_beta_p - real_test_results['precision']) / fixed_beta_p)
            buffered_test_precision_violation = max(0,
                                                    (fixed_beta_p - buffered_test_results[
                                                        'precision']) / fixed_beta_p)
            writer.writerow(
                ['final_test', real_test_results['accuracy'], real_test_results['precision'],
                 real_test_results['recall'],
                 buffered_test_results['accuracy'], buffered_test_results['precision'],
                 buffered_test_results['recall'], real_test_precision_violation, buffered_test_precision_violation])
        # full_mip
        gamma_0, z_plus_start, z_minus_start = initial_feasible_sol.calculate_gamma(obj_cons_num=obj_cons_num,
                                                                                    X_train=X_train,
                                                                                    y_train=y_train,
                                                                                    weights=w_start,
                                                                                    bias=b_start,
                                                                                    beta_p=full_mip_beta_p,
                                                                                    epsilon=epsilon)
        optimal_value, optimality_gap, optimal_w, optimal_b, optimal_z_plus, optimal_z_minus, objective_function_terms, real_results, buffered_results, counts_results = full_MIP.full_mip(
            model=model, obj_cons_num=obj_cons_num, X=X_train, y=y_train, w_start=w_start,
            b_start=b_start, z_plus_start=z_plus_start, z_minus_start=z_minus_start, epsilon=epsilon,
            gamma_0=gamma_0,
            M=M, rho=rho, beta_p=full_mip_beta_p, dirname=fixed_dirname)


AHC_optimization()
