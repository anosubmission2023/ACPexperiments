Temporary repository for the paper: Adaptive Conformal Prediction By Reweighting Nonconfirmity Score

**Adaptive Conformal Prediction** (ACP) is a Python package that aims to provide 
Adaptive Predictive Interval (PI) that better represent the uncertainty of the 
model by reweighting the NonConformal Score with the learned weights of a Random Forest.
 
## Requirements
Python 3.7+ 

**OSX**: ACP uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

Install the required packages:

```
$ pip install -r requirements.txt
```

## Installation

Clone the repo and run the following command in the ACP directory to install ACP
```
$ python setup.py install
```
You also need to install the package [PyGenStability](https://github.com/barahona-research-group/PyGenStability) that allows for community detection of the graph of the RF.

To make an all-in-one installation of all the setup for ACP and the SOTA to run the experiments of the paper, you can run the bash script: install.sh
```
$ bash install.sh
```

## Adaptive Conformal Predictions (ACP)
We propose 3 methods to compute PI: LCP-RF , LCP-RF-G, and QRF-TC. However, by default
we use QRF-TC as it is as accurate than the others while being more fast. The code of
LCP-RF and LCP-RF-G is not optimized yet.


**I. To compute PI using QRF-TC. First, we need to define ACXplainer module which has the same 
parameters as a classic RandomForest, so its parameters should be optimized to predict accurately the nonconfirmity score of the calibration datasets**
```python
from ACP.acv_explainers import ACXplainer

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acp = ACXplainer(classifier=False, n_estimators=100, mtry=int(np.sqrt(in_shape)),
                          max_depth=20, min_node_size=10)

acp.fit(x_calibration, v_calibration)

mae = mean_absolute_error(acv_xplainer.predict(x_calibration), v_calibration)
```

**II. Then, we can call the calibration method to run the training-conditional calibration.**

```python 
acp.fit_calibration(x_calibration, y_calibration, mean_estimator, quantile=1-alpha, only_qrf=True
                    score_type='mean')

# Use the code below for quantile score
# acp.fit_calibration(x_train[idx_cal], y_train[idx_cal], quantile_estimator, quantile=level, only_qrf=True
#                     score_type='quantile')
```

**II. Now, we can compute the Prediction Interval**
```python 
# You can compute directly the PI 
y_lower, y_upper = acp.predict_qrf_pi(x_test)

# Or use the code below to first compute the predicted residuals quantile, then compute the PI
# r['QRF_TC'] = acp.predict_qrf_r(x_test)
# y_lower['QRF_TC'] = pred_mean - r['QRF_TC']
# y_upper['QRF_TC'] = pred_mean + r['QRF_TC']

```



## Notebooks

The notebook below show how to you use ACP in practice.
- HOW_TO_ACP.ipynb

The following notebook permit to re-run the experiments of the paper, but before you
need to run the bash script: install.sh to install the SLCP package.

- EXAMPLE_mean_estimate_pi.ipynb
- RUN_paper_experiments_mean_score.ipynb
- RUN_paper_experiments_quantile_score.ipynb


**Remaks**: The ACP package is currently included within a large eXplainable AI (XAI) package, as our initial goal was to use XAI tools to explain the QRF used to learn the nonconformity score, in order to provide a clear and interpretable description of the regions (rules) of high/low uncertainty.