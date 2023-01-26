import pandas as pd
import os
import numpy as np
from utils import plot_func
from SLCP.datasets import datasets
from SLCP import config
from cqr import helper

save_path = '/home/xxxxx/saved_co/'

dataset_name = 'star'
model_name = 'random_forest'

estimator_map = {'SPLIT': 'mean', 'SLCP': 'mean', 'CQR': 'quantile', 'LCP-RF': 'mean', 'LCP-RF-Group': 'mean',
                 'QRF_train': 'mean', 'LCP': 'mean'}
color_map = {'SPLIT': 'red', 'SLCP': 'red', 'CQR': 'red', 'LCP-RF': 'red', 'LCP-RF-Group': 'red',
             'QRF_train': 'red', 'LCP': 'red'}
model_map = {'random_forest': 'Random Forest', 'linear': 'Linear Regression', 'neural_net': 'Neural Network',
             'kde': 'Constant', 'xgboost':'XGBRegressor'}


method = 'LCP'
y_lower = {}
y_upper = {}
coverages = {}
lengths = {}
lengths_residuals = {}

save_path = os.path.join('/home/xxxxx/saved_co/', dataset_name)
data_path = os.path.join(save_path, model_name)


r_lcp = np.array(pd.read_csv(os.path.join(data_path, 'lcp_residuals.csv')).x.values).reshape(-1)
x_test = pd.read_csv(os.path.join(data_path, 'x_test_gen.csv'), header=None).values
v_test = np.array(pd.read_csv(os.path.join(data_path, 'y_test_gen.csv'), header=None).values).reshape(-1)
y_test = np.array(pd.read_csv(os.path.join(data_path, 'y_test.csv'), header=None).values).reshape(-1)
pred = np.array(pd.read_csv(os.path.join(data_path, 'test_pred.csv'), header=None).values).reshape(-1)
y_upper = pd.read_csv(os.path.join(data_path, 'y_upper' + '_' + model_name + '_' + dataset_name + '.csv'), index_col=0)
y_lower = pd.read_csv(os.path.join(data_path, 'y_lower' + '_' + model_name + '_' + dataset_name + '.csv'), index_col=0)

y_lower[method] = pred - r_lcp
y_upper[method] = pred + r_lcp

plot_func(x=x_test[:, 0].reshape(-1, 1),
                      y=y_test,
                      y_u=y_upper[method],
                      y_l=y_lower[method],
                      pred=None,
                      shade_color=color_map[method],
                      method_name=method + ":",
                      title=f"{method} {model_map[model_name]} ({estimator_map[method]} regression)",
                      filename=os.path.join(data_path, method + '_' + model_name + '_' + dataset_name + '.png'),
                      save_figures=config.UtilsParams.save_figures,
                      split_upper=y_upper['SPLIT'],
                      split_lower=y_lower['SPLIT'])

coverages[method], lengths[method] = helper.compute_coverage(y_test, y_lower[method], y_upper[method],
                                                             config.ConformalParams.alpha, method)

lengths_residuals[method] = np.mean(np.abs(v_test - r_lcp))

metrics = pd.read_csv(os.path.join(data_path, 'metrics' + '_' +model_name + '_' + dataset_name + '.csv'), index_col=0)

lcp_metrics = {"coverages":coverages[method], "lengths":lengths[method],
               'lengths_residuals':lengths_residuals[method]}

metrics_all = metrics.append(pd.DataFrame([lcp_metrics], index=['LCP'], columns=metrics.columns))
metrics_all.to_csv(os.path.join(data_path, 'all_metrics' + '_' + model_name + '_' + dataset_name + '.csv'))
