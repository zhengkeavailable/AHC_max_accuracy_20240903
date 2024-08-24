import train_and_evaluate
import epsilon_shrinkage
import fixed_epsilon
import initialization
import csv


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
         max_inner_iteration, max_outer_iteration, max_fixed_iteration, gap, sigma, base_rate, enlargement_rate,
         shrinkage_rate,
         pip_max_rate,
         outer_dirname, fixed_dirname,
         outer_result_file, fixed_result_file, full_mip) = initialization.initialization(positive_size=para[0],
                                                                                         negative_size=para[1],
                                                                                         outer_beta_p=min(para[2], 1),
                                                                                         fixed_beta_p=min(para[3], 1),
                                                                                         file_path=para[4],
                                                                                         full_mip=para[5])
        # if not full_mip:
        outer_objective_value, outer_weight, outer_bias, outer_z_plus, outer_z_minus, outer_precision_in_constraint = epsilon_shrinkage.epsilon_shrinkage(
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
        fixed_objective_value, fixed_weight, fixed_bias, fixed_z_plus, fixed_z_minus, fixed_precision_in_constraint = fixed_epsilon.fixed_epsilon(
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
        # if full mip: full_MIP.full_mip()
        with open(outer_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test,
                                                                                                  outer_weight,
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
            real_test_results, buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test,
                                                                                                  fixed_weight,
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
