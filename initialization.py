import train_and_evaluate
import precision_and_accuracy_curve
import gurobipy as gp
import os
from datetime import datetime
import csv


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
    results = train_and_evaluate.train_and_evaluate(file_path, n_iters, positive_size=positive_size,
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
    initial_real_train_results, initial_buffered_train_results = train_and_evaluate.evaluate_classification(X_train, y_train,
                                                                                                     w_start,
                                                                                                     b_start)
    initial_real_test_results, initial_buffered_test_results = train_and_evaluate.evaluate_classification(X_test, y_test,
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
    precision_and_accuracy_curve.precision_accuracy_curve(X_train, results['y_train'], w_start, b_start, num_thresholds=100,
                                         save_path=outer_dirname + '/outer_precision_accuracy_curve_initial_train_beta_p=' + str(
                                             outer_beta_p) + '.png')
    precision_and_accuracy_curve.precision_accuracy_curve(X_test, results['y_test'], w_start, b_start, num_thresholds=100,
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
    precision_and_accuracy_curve.precision_accuracy_curve(X_train, results['y_train'], w_start, b_start, num_thresholds=100,
                                         save_path=fixed_dirname + '/fixed_precision_accuracy_curve_initial_train_beta_p=' + str(
                                             fixed_beta_p) + '.png')
    precision_and_accuracy_curve.precision_accuracy_curve(X_test, results['y_test'], w_start, b_start, num_thresholds=100,
                                         save_path=fixed_dirname + '/fixed_precision_accuracy_curve_initial_test_beta_p=' + str(
                                             fixed_beta_p) + '.png')
    return (
        model, obj_cons_num, X_train, y_train, X_test, y_test, w_start, b_start, epsilon, M, rho, fixed_beta_p,
        outer_beta_p, lbd, lbd_2,
        max_inner_iteration, max_outer_iteration, max_fixed_iteration,
        gap, sigma, base_rate, enlargement_rate, shrinkage_rate, pip_max_rate, outer_dirname, fixed_dirname,
        outer_result_file, fixed_result_file, full_mip)