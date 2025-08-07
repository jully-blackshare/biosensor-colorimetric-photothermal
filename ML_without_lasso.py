# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.utils import resample 
import matplotlib.pyplot as plt 

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

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_lin_pred = lin_model.predict(X_test)

y_pred_lin = 10**(y_lin_pred)
y_actual_lin = 10**y_test

# Polynomial Regression
degree = 2
poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly.fit(X_train, y_train)
y_poly_pred = poly.predict(X_test)

# Sigmoid Regression
def sigmoid_function(X, a1, a2, a3, a4, a5):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return a1 / (1 + np.exp(-(a2 * x1 + a3 * x2 + a4 * x3 + a5)))

X_train_array = X_train.values
y_train_array= y_train.values
initial_guess = [max(y_train), 0.5, 0.5, 0.5, 0]

sigmoid = lambda X1, a1, a2, a3, a4, a5: sigmoid_function(np.column_stack((X1[0], X1[1], X1[2])), a1, a2, a3, a4, a5)

X_components = [X_train_array[:, i] for i in range(3)]
popt, _ = curve_fit(sigmoid, X_components, y_train_array, p0=initial_guess, maxfev=10000)

X_test_array = X_test.values
y_sig_pred = sigmoid_function(X_test_array, *popt)

def boostrap_prediction_confidence_intervals(model, X_train, y_train, X_test, n_iter = 1000, alpha = 0.05):
    predictions = np.zeros((len(X_test), n_iter))

    for i in range(n_iter):
        X_resample, y_resample = resample(X_train, y_train)
        mdl = model()
        mdl.fit(X_resample, y_resample)
        predictions[:, i] = mdl.predict(X_test)

    lower = np.percentile(predictions, 100 * (alpha/2) , axis = 1)
    upper = np.percentile(predictions, 100 * (1-alpha/2), axis = 1)
    return lower, upper

def bootstrap_pseudo_r2(model, X, y, n_iter = 1000):
    pseudo_r2 = []

    for _ in range(n_iter):
        X_resample, y_resample = resample(X, y)
        X_train_boot, X_test_boot, y_train_boot, y_test_boot = train_test_split(X_resample, y_resample, test_size = 0.2)
        mdl = model()
        mdl.fit(X_train_boot, y_train_boot)
        y_pred = mdl.predict(X_test_boot)

        logL_model = -0.5*len(y_test_boot) * np.log(np.mean((y_test_boot - y_pred) ** 2))
        logL_null = -0.5 * len(y_test_boot) * np.log(np.mean((y_test_boot - np.mean(y_test_boot)) ** 2))
        p_r2 = 1 - (logL_model / logL_null)

        pseudo_r2.append(p_r2)

    lower = np.percentile(pseudo_r2, 2.5)
    upper = np.percentile(pseudo_r2, 97.5)
    return lower, upper, pseudo_r2

def get_linear_regression():
    return LinearRegression()

def get_poly_regression():
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

sigmoid_preds = np.zeros((len(X_test), 1000))
for i in range(1000):
    X_resample, y_resample = resample(X_train_array, y_train_array)
    try:
        popt_i, _ = curve_fit(sigmoid, [X_resample[:,0], X_resample[:,1], X_resample[:,2]], y_resample, p0=initial_guess, maxfev=10000)
        pred_i = sigmoid_function(X_test_array, *popt_i)
        sigmoid_preds[:, i] = pred_i
    except RuntimeError:
        sigmoid_preds[:, i] = np.nan  # handle failed fits

y_predictions = [y_lin_pred, y_poly_pred, y_sig_pred]

pseudo_r2 = []
for pred in y_predictions:
    logL_model = -0.5 * len(y_test) * np.log(np.mean((y_test - pred) ** 2))
    logL_null = -0.5 * len(y_test) * np.log(np.mean((y_test - np.mean(y_test)) ** 2))
    p_r2 = 1 - (logL_model/logL_null)
    pseudo_r2.append(p_r2)

# caluclate AIC and BIC
def compute_aic_bic(y_true, y_pred, k):
    n = len(y_true)
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    log_likelihood = -0.5 * n * np.log(mse)

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

"""lin_order = X_train.shape[1] + 1
poly_order = poly.named_steps['lassocv'].coef_.shape[0] + 1
sig_order = 5

aic_lin, bic_lin = compute_aic_bic(y_test, y_lin_pred, lin_order)
aic_poly, bic_poly = compute_aic_bic(y_test, y_poly_pred, poly_order)
aic_sig, bic_sig = compute_aic_bic(y_test, y_sig_pred, sig_order)"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score

# Define a scoring function (e.g., R²)
scorer = make_scorer(r2_score)

# Linear Regression Cross-validation
lin_scores = cross_val_score(get_linear_regression(), X, y, cv=5, scoring=scorer)
print(f"Linear Regression CV R²: Mean = {np.mean(lin_scores):.4f}, Std = {np.std(lin_scores):.4f}")

# Polynomial Regression Cross-validation
poly_scores = cross_val_score(get_poly_regression(), X, y, cv=5, scoring=scorer)
print(f"Polynomial Regression CV R²: Mean = {np.mean(poly_scores):.4f}, Std = {np.std(poly_scores):.4f}")


from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np

# Assuming X and y are already defined in your script
X_vals = X.values.flatten()
y_vals = y.values.flatten()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []

for train_idx, test_idx in kf.split(X_vals, y_vals):
    X_train, X_test = X_vals[train_idx], X_vals[test_idx]
    y_train, y_test = y_vals[train_idx], y_vals[test_idx]
    
    # Initial guess (you probably already use this)
    p0 = [max(y_train), np.median(X_train), 1, min(y_train)]
    
    try:
        popt, _ = curve_fit(sigmoid, X_train, y_train, p0, maxfev=10000)
        y_pred = sigmoid(X_test, *popt)
        
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    except RuntimeError:
        print("Fit failed for a fold.")
        continue

print(f"Cross-validated R² (Sigmoid): Mean = {np.mean(r2_scores):.4f}, Std = {np.std(r2_scores):.4f}")
