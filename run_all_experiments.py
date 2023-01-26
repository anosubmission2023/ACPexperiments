import os
from SLCP import config
import numpy as np
import pandas as pd
from SLCP.datasets import datasets
from conformal import ConformalPred
from utils import plot_pred, set_seed, plot_func
from SLCP.cqr import helper
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from xgboost import XGBRegressor
from acv_explainers.utils import mean_score, quantile_score
import logging
import copy
import sys
from sklearn.metrics import mean_absolute_error
from acv_explainers.utils import save_model
from tqdm import tqdm
from acv_explainers import ACXplainer


logger = logging.getLogger('SLCP.prediction')
base_dataset_path = '/home/xxxxx/acp_experiments/SLCP/datasets/'


def run_pred_experiment(dataset_name, model_name, random_seed, base_path, estimator='mean'):
    set_seed(random_seed)
    try:
        X_train, X_test, y_train, y_test = datasets.GetDataset(dataset_name,
                                                               base_dataset_path,
                                                               random_seed,
                                                               config.DataParams.test_ratio)
    except:
        logger.info("CANNOT LOAD DATASET!")
        return
    x_train, x_test = X_train.astype(np.double)[:config.DataParams.max_train], X_test.astype(np.double)[:config.DataParams.max_test]
    y_train, y_test = y_train.astype(np.double).reshape(-1)[:config.DataParams.max_train], y_test.astype(np.double).reshape(-1)[:config.DataParams.max_test]

    if 'simulation' in dataset_name or 'shift' in dataset_name:
        n_train = config.DataParams.n_train
        # augmented 1-D data into 2-D  for ACVTree
        x_te, x_ts = np.zeros(shape=(x_train.shape[0], 2)), np.zeros(shape=(x_test.shape[0], 2))
        x_te[:, 0], x_te[:, 1] = x_train[:, 0], x_train[:, 0]
        x_ts[:, 0], x_ts[:, 1] = x_test[:, 0], x_test[:, 0]
        x_train = x_te.astype(np.float64)
        x_test = x_ts.astype(np.float64)
    else:
        n_train = x_train.shape[0]

    in_shape = x_train.shape[1]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * config.ConformalParams.valid_ratio))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]

    save_path = os.path.join(base_path, dataset_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if model_name == 'random_forest':
        mean_estimator = RandomForestRegressor(n_estimators=config.RandomForecastParams.n_estimators,
                                               min_samples_leaf=config.RandomForecastParams.min_samples_leaf,
                                               max_features=in_shape,
                                               random_state=config.RandomForecastParams.random_state)
        quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                                   fit_params=None,
                                                                   quantiles=config.ConformalParams.quantiles,
                                                                   params=config.RandomForecastParams)

    elif model_name == 'xgboost':
        mean_estimator = XGBRegressor(learning_rate =0.01,
                                     n_estimators=1000,
                                     max_depth=10,
                                     min_child_weight=6,
                                     gamma=0,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     reg_alpha=0.005,
                                     nthread=4,
                                     scale_pos_weight=1,
                                     seed=27)
        quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                                   fit_params=None,
                                                                   quantiles=config.ConformalParams.quantiles,
                                                                   params=config.RandomForecastParams)


    elif model_name == 'linear':
        mean_estimator = helper.MSELR_RegressorAdapter(model=None,
                                                       in_shape=in_shape,
                                                       epochs=config.LinearParams.epochs,
                                                       lr=config.LinearParams.lr,
                                                       batch_size=config.LinearParams.batch_size,
                                                       wd=config.LinearParams.wd,
                                                       test_ratio=config.LinearParams.test_ratio,
                                                       random_state=config.LinearParams.random_state)

        quantile_estimator = helper.QLR_RegressorAdapter(model=None,
                                                         in_shape=in_shape,
                                                         epochs=config.LinearParams.epochs,
                                                         lr=config.LinearParams.lr,
                                                         batch_size=config.LinearParams.batch_size,
                                                         wd=config.LinearParams.wd,
                                                         test_ratio=config.LinearParams.test_ratio,
                                                         random_state=config.LinearParams.random_state)

    elif model_name == 'neural_net':

        mean_estimator = helper.MSENet_RegressorAdapter(model=None,
                                                        in_shape=in_shape,
                                                        hidden_size=config.NeuralNetParams.hidden_size,
                                                        epochs=config.NeuralNetParams.epochs,
                                                        lr=config.NeuralNetParams.lr,
                                                        batch_size=config.NeuralNetParams.batch_size,
                                                        dropout=config.NeuralNetParams.dropout,
                                                        wd=config.NeuralNetParams.wd,
                                                        test_ratio=config.NeuralNetParams.test_ratio,
                                                        random_state=config.NeuralNetParams.random_state)

        quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                             in_shape=in_shape,
                                                             hidden_size=config.NeuralNetParams.hidden_size,
                                                             epochs=config.NeuralNetParams.epochs,
                                                             lr=config.NeuralNetParams.lr,
                                                             batch_size=config.NeuralNetParams.batch_size,
                                                             dropout=config.NeuralNetParams.dropout,
                                                             wd=config.NeuralNetParams.wd,
                                                             test_ratio=config.NeuralNetParams.test_ratio,
                                                             random_state=config.NeuralNetParams.random_state)

    elif model_name == 'kde':
        mean_estimator = helper.MSEConst_RegressorAdapter()

        quantile_estimator = helper.QConst_RegressorAdapter()

    # METRICS
    coverages = {}
    lengths = {}
    lengths_residuals = {}
    y_lower = {}
    y_upper = {}
    r = {}

    methods = ['SPLIT', 'SPLIT-G', 'SLCP', 'CQR', 'LCP-RF', 'LCP-RF-Group', 'QRF_train']

    mean_estimator.fit(x_train[idx_train], y_train[idx_train])
    quantile_estimator.fit(x_train[idx_train], y_train[idx_train])

    pred_mean = mean_estimator.predict(x_test).astype(np.float64)
    test_residuals = mean_score(mean_estimator.predict(x_test), y_test)
    test_quantile_residuals = quantile_score(quantile_estimator.predict(x_test), y_test)

    # saved for data for LCP
    v_train = np.abs(mean_estimator.predict(x_train[idx_train]) - y_train[idx_train])
    v_cal = np.abs(mean_estimator.predict(x_train[idx_cal]) - y_train[idx_cal])
    v_test = np.abs(mean_estimator.predict(x_test) - y_test)

    data_path = os.path.join(save_path, model_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    np.savetxt(os.path.join(data_path, 'x_train_gen.csv'), x_train[idx_train], delimiter=",")
    np.savetxt(os.path.join(data_path, "y_train_gen.csv"), v_train, delimiter=",")

    np.savetxt(os.path.join(data_path, "x_validation_gen.csv"), x_train[idx_cal], delimiter=",")
    np.savetxt(os.path.join(data_path, "y_validation_gen.csv"), v_cal, delimiter=",")

    np.savetxt(os.path.join(data_path, "x_test_gen.csv"), x_test, delimiter=",")
    np.savetxt(os.path.join(data_path, "y_test_gen.csv"), v_test, delimiter=",")

    np.savetxt(os.path.join(data_path, 'test_pred.csv'), pred_mean, delimiter=",")
    np.savetxt(os.path.join(data_path, 'y_test.csv'), y_test, delimiter=",")

    # SPLIT conformal
    cp_split = ConformalPred(model=mean_estimator,
                             method='split',
                             data_name=dataset_name,
                             ratio=config.ConformalParams.valid_ratio,
                             x_train=x_train,
                             x_test=x_test,
                             y_train=y_train,
                             y_test=y_test,
                             k=config.ConformalParams.k)

    cp_split.idx_train = idx_train
    cp_split.idx_cal = idx_cal
    cp_split.fit()

    y_lower_split, y_upper_split = cp_split.predict()
    y_lower['SPLIT'] = y_lower_split
    y_upper['SPLIT'] = y_upper_split
    r['SPLIT'] = y_upper_split - y_lower_split

    # SLCP conformal
    if estimator == 'mean':
        cp_slcp = ConformalPred(model=mean_estimator,
                                method='slcp-mean',
                                data_name=dataset_name,
                                ratio=config.ConformalParams.valid_ratio,
                                x_train=x_train,
                                x_test=x_test,
                                y_train=y_train,
                                y_test=y_test,
                                k=config.ConformalParams.k)
    else:
        cp_slcp = ConformalPred(model=quantile_estimator,
                                method='slcp-rbf',
                                data_name=dataset_name,
                                ratio=config.ConformalParams.valid_ratio,
                                x_train=x_train,
                                x_test=x_test,
                                y_train=y_train,
                                y_test=y_test,
                                k=config.ConformalParams.k)

    cp_slcp.idx_train = idx_train
    cp_slcp.idx_cal = idx_cal
    cp_slcp.fit()


    y_lower_slcp, y_upper_slcp = cp_slcp.predict()
    y_lower['SLCP'] = y_lower_slcp
    y_upper['SLCP'] = y_upper_slcp
    r['SLCP'] = y_upper_slcp - y_lower_slcp

    # CQR NS
    cp_cqr = ConformalPred(model=quantile_estimator,
                           method='cqr',
                           data_name=dataset_name,
                           ratio=config.ConformalParams.valid_ratio,
                           x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test,
                           k=config.ConformalParams.k)
    cp_cqr.idx_train = idx_train
    cp_cqr.idx_cal = idx_cal
    cp_cqr.fit()
    y_lower_cqr, y_upper_cqr = cp_cqr.predict()
    y_lower['CQR'] = y_lower_cqr
    y_upper['CQR'] = y_upper_cqr
    r['CQR'] = y_upper_cqr - y_lower_cqr

    # LCP-RF
    level = 1 - config.ConformalParams.alpha
    acv_xplainer = ACXplainer(classifier=False, n_estimators=200,
                              mtry=int(in_shape),
                              max_depth=20,
                              min_node_size=10,
                              seed=config.RandomForecastParams.random_state)

    acv_xplainer.fit_calibration(x_train[idx_cal], y_train[idx_cal], mean_estimator,
                                 bygroup=True,
                                 training_conditional=True, training_conditional_one=True,
                                 quantile=level)

    r['LCP-RF'], s_lcp_all = acv_xplainer.predict_rf_lcp_train_one(x_test, level)
    y_lower['LCP-RF'] = pred_mean - r['LCP-RF']
    y_upper['LCP-RF'] = pred_mean + r['LCP-RF']

    r['LCP-RF-Group'], y_lcp_group = acv_xplainer.predict_rf_lcp_bygroup_train(x_test, level)
    y_lower['LCP-RF-Group'] = pred_mean - r['LCP-RF-Group']
    y_upper['LCP-RF-Group'] = pred_mean + r['LCP-RF-Group']

    r['QRF_train'] = acv_xplainer.predict_qrf_r(x_test)
    y_lower['QRF_train'] = pred_mean - r['QRF_train']
    y_upper['QRF_train'] = pred_mean + r['QRF_train']

    acv = acv_xplainer
    y_test_base = np.zeros(x_test.shape[0])
    w_test = acv.compute_forest_weights(x_test, y_test_base, acv.x_cali, acv.r_cali)
    groups = np.unique(acv.communities)
    p_test = np.zeros(shape=(x_test.shape[0], len(groups)))
    for group in groups:
        p_test[:, group] = np.sum(w_test[:, acv.communities == group], axis=1)

    group_test = np.argmax(p_test, axis=1)

    n_test = x_test.shape[0]
    r_split_group = np.zeros(n_test)

    print('Test Groupwise')
    r_cali = acv.r_cali
    for i in tqdm(range(n_test)):
        group_idx = acv.communities == group_test[i]

        n_group = np.sum(group_idx)
        r_candidate = r_cali[group_idx]

        k_star = np.int((n_group + 1) * (level))
        r_split_group[i] = np.sort(r_candidate)[k_star-1]
        # r_split_group[i] = np.max(r_cali)
        # print(r_split_group[i], y_test[i])
    y_lower['SPLIT-G'] = pred_mean - r_split_group
    y_upper['SPLIT-G'] = pred_mean + r_split_group
    r['SPLIT-G'] = r_split_group

    estimator_map = {'SPLIT': 'mean', 'SLCP': 'mean', 'CQR': 'quantile', 'LCP-RF': 'mean', 'LCP-RF-Group': 'mean',
                     'QRF_train': 'mean', 'SPLIT-G':'mean'}
    color_map = {'SPLIT': 'red', 'SLCP': 'red', 'CQR': 'red', 'LCP-RF': 'red', 'LCP-RF-Group': 'red',
                 'QRF_train': 'red', 'SPLIT-G':'red'}
    model_map = {'random_forest': 'Random Forest', 'linear': 'Linear Regression', 'neural_net': 'Neural Network',
                 'kde': 'Constant', 'xgboost': 'XGBRegressor'}

    for method in methods:
        coverages[method], lengths[method] = helper.compute_coverage(y_test, y_lower[method], y_upper[method],
                                                                     config.ConformalParams.alpha, method)
        if estimator_map[method] == 'mean':
            lengths_residuals[method] = np.mean(np.abs(test_residuals - r[method]))
        else:
            lengths_residuals[method] = np.mean(np.abs(test_quantile_residuals - r[method]))

    metrics_data = [pd.DataFrame.from_dict(coverages, orient='index', columns=['coverages']),
                    pd.DataFrame.from_dict(lengths, orient='index', columns=['lengths']),
                    pd.DataFrame.from_dict(lengths_residuals, orient='index', columns=['lengths_residuals'])]

    metrics = pd.concat(metrics_data, axis=1)
    metrics.to_csv(os.path.join(data_path, 'metrics' + '_' + model_name + '_' + dataset_name + '.csv'))

    r_data = pd.DataFrame(r)
    r_data.to_csv(os.path.join(data_path, 'r_all' + '_' + model_name + '_' + dataset_name + '.csv'))

    y_lower_data = pd.DataFrame(y_lower)
    y_lower_data.to_csv(os.path.join(data_path, 'y_lower' + '_' + model_name + '_' + dataset_name + '.csv'))

    y_upper_data = pd.DataFrame(y_upper)
    y_upper_data.to_csv(os.path.join(data_path, 'y_upper' + '_' + model_name + '_' + dataset_name + '.csv'))

    if 'simulation' in dataset_name or 'gen' in dataset_name or "shift" in dataset_name:
        for method in methods:
            plot_func(x=x_test[:, 0].reshape(-1, 1),
                      y=y_test,
                      y_u=y_upper[method],
                      y_l=y_lower[method],
                      pred=None,
                      shade_color=color_map[method],
                      method_name=method + ":",
                      title=f"{method} {model_map[model_name]} ({estimator_map[method]} regression)",
                      filename=os.path.join(data_path, method + '_' + model_name + '_' + dataset_name + '.pdf'),
                      save_figures=config.UtilsParams.save_figures,
                      split_lower=y_lower_split,
                      split_upper=y_upper_split)

    save_model(acv_xplainer, os.path.join(save_path, dataset_name + '_' + model_name + '_' + dataset_name))

    return metrics


run_pred_experiment('bike', 'linear', random_seed=2022, base_path='/home/xxxxx/saved_acp/', estimator='mean')
