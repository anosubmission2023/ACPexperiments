import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skranger.ensemble import RangerForestRegressor
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from acv_explainers import ACXplainer
np.random.seed(2022)
X, y = make_regression(n_informative=5, n_samples=1000, n_features=50, noise=0.1, random_state=59)
# X, y = shap.datasets.diabetes()
# X, y = X.values, y

X_val, y_val = X[:100], y[:100]
X_train, X_test, y_train, y_test = train_test_split(X[100:], y[100:], test_size=0.33, random_state=2022)

regressor = XGBRegressor()
regressor.fit(X_train, y_train)
mean_absolute_error(regressor.predict(X_test), y_test)

R_test = np.abs(regressor.predict(X_test) - y_test)
sorter = np.argsort(R_test)

acv_xplainer = ACXplainer(classifier=False, n_estimators=20, max_depth=10, min_node_size=5, replace=True)
acv_xplainer.fit(X_test, R_test)

S = list(range(X_test.shape[1]))

S_val = X_test.shape[0] * [S]
_, w = acv_xplainer.compute_quantile_rf(X_test, R_test, X_test, R_test, S=S_val, quantile=90, min_node_size=5,
                                       min_node_size_intersection=0)