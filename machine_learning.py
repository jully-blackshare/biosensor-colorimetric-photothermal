import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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

k_lin = X_train.shape[1]+1
k_poly = PolynomialFeatures(degree=2).fit(X_train).n_output_features_
k_sig = 5

aic_lin, bic_lin = compute_aic_bic(y_test, y_lin_pred, k_lin)
aic_poly, bic_poly = compute_aic_bic(y_test, y_poly_pred, k_poly)
aic_sig, bic_sig = compute_aic_bic(y_test, y_sig_pred, k_sig)

print(pseudo_r2)
print(aic_sig, bic_sig)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {
    'linear': {'r2': [], 'aic': [], 'bic': []},
    'poly': {'r2': [], 'aic': [], 'bic': []},
    'sigmoid': {'r2': [], 'aic': [], 'bic': []},
}

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- Linear Regression ---
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_lin_pred = lin_model.predict(X_test)

    logL_model_lin = -0.5 * len(y_test) * np.log(np.mean((y_test - y_lin_pred) ** 2))
    logL_null_lin = -0.5 * len(y_test) * np.log(np.mean((y_test - np.mean(y_test)) ** 2))
    pseudo_r2_lin = 1 - (logL_model_lin / logL_null_lin)
    results['linear']['r2'].append(pseudo_r2_lin)

    k_lin = X_train.shape[1] + 1
    aic_lin, bic_lin = compute_aic_bic(y_test, y_lin_pred, k_lin)
    results['linear']['aic'].append(aic_lin)
    results['linear']['bic'].append(bic_lin)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # --- Polynomial Regression ---
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_poly_pred = poly_model.predict(X_test)

    logL_model_poly = -0.5 * len(y_test) * np.log(np.mean((y_test - y_poly_pred) ** 2))
    logL_null_poly = -0.5 * len(y_test) * np.log(np.mean((y_test - np.mean(y_test)) ** 2))
    pseudo_r2_poly = 1 - (logL_model_poly / logL_null_poly)
    results['poly']['r2'].append(pseudo_r2_poly)
    
    k_poly = PolynomialFeatures(degree=2).fit(X_train).n_output_features_
    aic_poly, bic_poly = compute_aic_bic(y_test, y_poly_pred, k_poly)
    results['poly']['aic'].append(aic_poly)
    results['poly']['bic'].append(bic_poly)

    # --- Sigmoid Regression ---
try:
    from scipy.optimize import curve_fit

    def sigmoid(x, a1, a2, a3, a4, a5):
        return a1 / (1 + np.exp(-a2 * (x @ np.array([a3, a4, a5]))))

    X_train_vals = X_train.values
    X_test_vals = X_test.values
    y_train_vals = y_train.values
    y_test_vals = y_test.values

    # Initial guess for parameters
    p0 = [max(y_train_vals), 1, 1, 1, min(y_train_vals)]
    popt, _ = curve_fit(sigmoid, X_train_vals, y_train_vals, p0, maxfev=10000)
    y_sig_pred = sigmoid(X_test_vals, *popt)

    # --- Pseudo R² ---
    logL_model = -0.5 * len(y_test) * np.log(np.mean((y_test - y_lin_pred) ** 2))
    logL_null = -0.5 * len(y_test) * np.log(np.mean((y_test - np.mean(y_test)) ** 2))
    pseudo_r2 = 1 - (logL_model / logL_null)
    results['sigmoid']['r2'].append(pseudo_r2)

    # --- AIC/BIC ---
    k_sig = 5  # number of parameters in sigmoid function
    aic, bic = compute_aic_bic(y_test_vals, y_sig_pred, k_sig)
    results['sigmoid']['aic'].append(aic)
    results['sigmoid']['bic'].append(bic)

except RuntimeError:
    print("Sigmoid fit failed in one fold. Skipping.")

print(results)