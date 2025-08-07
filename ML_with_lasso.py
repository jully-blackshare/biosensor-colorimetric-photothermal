import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

file_paths = {
    "10E4": "path/to/10E4_augmented.xlsx",
    "10E5": "pth/to/10E5_augmented.xlsx",
    "10E6": "path/to/10E6_augmented.xlsx",
    "10E7": "path/to/10E7_augmented.xlsx",
}

dfs = [pd.read_excel(path) for path in file_paths.values()]
df = pd.concat(dfs, ignore_index=True)

df = df[['Red Ratio', 'Green Ratio', 'Blue Ratio', 'Target']].copy()
df['log_red'] = np.log(df['Red Ratio'])
df['log_green'] = np.log(df['Green Ratio'])
df['log_blue'] = np.log(df['Blue Ratio'])
df['Target_exp'] = 10 ** df['Target']

X = df[['log_red', 'log_green', 'log_blue']]
y = df['Target_exp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Polynomial regression
degree = 2
poly = make_pipeline(PolynomialFeatures(degree), LassoCV(cv=5, max_iter=10000))

poly.fit(X_train, y_train)
y_poly_pred = poly.predict(X_test)

# Lasso 
lasso = LassoCV(cv=5).fit(X_train, y_train)
y_lasso_pred = lasso.predict(X_test)

# Sigmoid Regression
def sigmoid_function(X, a1, a2, a3, a4, a5):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return a1 / (1 + np.exp(-(a2 * x1 + a3 * x2 + a4 * x3 + a5)))

# L1 function for sigmoid model 
def l1_loss(params, X, y, lambda_):
    y_pred = sigmoid_function(X, *params)
    mse = np.mean((y-y_pred)**2)
    l1_penalty = lambda_ * np.sum(np.abs(params[1:]))
    return mse + l1_penalty

X_train_array = X_train[['log_red', 'log_green', 'log_blue']].values
y_train_array = y_train.values
initial_guess = [max(y_train), 0.5, 0.5, 0.5, 0]
lambda_ = 0.001

result = minimize(
    l1_loss,
    x0 = initial_guess,
    args = (X_train_array, y_train_array, lambda_),
    method='L-BFGS-B'
)

popt_l1 = result.x
X_test_array = X_test[['log_red', 'log_green', 'log_blue']].values
y_sig_pred = sigmoid_function(X_test_array, *popt_l1)

y_predictions = [y_lasso_pred, y_poly_pred, y_sig_pred]

pseudo_r2 = []
for pred in y_predictions:
    logL_model = -0.5 * len(y_test) * np.log(np.mean((y_test - pred) ** 2))
    logL_null = -0.5 * len(y_test) * np.log(np.mean((y_test - np.mean(y_test)) ** 2))
    p_r2 = 1 - (logL_model/logL_null)
    pseudo_r2.append(p_r2)

# calculating AIC and BIC
def compute_aic_bic(y_true, y_pred, k):
    n = len(y_true)
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    log_likelihood = -0.5 * n * np.log(mse)

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

lin_order = X_train.shape[1] + 1
poly_order = poly.named_steps['lassocv'].coef_.shape[0] + 1
sig_order = 5

aic_lin, bic_lin = compute_aic_bic(y_test, y_lasso_pred, lin_order)
aic_poly, bic_poly = compute_aic_bic(y_test, y_poly_pred, poly_order)
aic_sig, bic_sig = compute_aic_bic(y_test, y_sig_pred, sig_order)