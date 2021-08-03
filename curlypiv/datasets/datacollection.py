"""
Notes about this program:

"""

# import modules
import numpy as np
from numpy import mean, std, absolute

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


from curlypiv.utils.read_XML import read_Squires_XML

import matplotlib.pyplot as plt


# step 1: import squires dataset to Pandas DataFrame
save_squires = False
save_squires_path = '/Users/mackenzie/PythonProjects/curlypiv-master/curlypiv/data/results/squires_data.csv'
df = read_Squires_XML(write_to_disk=save_squires, write_path=save_squires_path)
print(df.head(5))

# step 2: make all values numeric and reorganize columns
df.loc[(df.dielectrics == 'SiO2'), 'dielectrics'] = 1
df.loc[(df.buffers == 'KCl'), 'buffers'] = 1
df.loc[(df.buffers == 'NaCl'), 'buffers'] = 0
u_max = df.raw_uvel_max.values
df.insert(loc=0, column='u_max', value=u_max)
df.drop(labels='raw_uvel_max', axis=1, inplace=True)

# assign label and features
data = df.values
X, y = data[:, 1:], data[:, 0]

# define training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = ElasticNet(alpha=1e-6, l1_ratio=0.99)
model.fit(X_train, y_train)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) # repeated 10-fold cross-validation
scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = absolute(scores) # force scores to be positive
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
print("MAE: mean absolute error")

# print model prediction as a function of u_max
y_pred = model.predict(X_test)

# convert to correct datatypes
y_test = y_test.astype(float)
y_pred = y_pred.astype(float)

# plot results
plt.figure()
plt.scatter(y_test, y_pred, marker='.', label='Prediction (Mean MAE: %.3f)' % (mean(scores)))
line = np.linspace(np.min(y_test), np.max(y_test))
plt.plot(line, line, color='gray', alpha=0.5, label='True')
plt.xlabel(r'$U_{slip, max}$  $\left(\frac {\mu m}{s}\right)$')
plt.ylabel(r'$U_{slip, max}$  $\left(\frac {\mu m}{s}\right)$')
plt.title(r'$ElasticNet$ Prediction Accuracy on Squires Dataset')
plt.legend()
plt.show()

# plot error scaled by max velocity
y_norm_error = (y_pred-y_test)/y_test
plt.figure()
plt.scatter(y_test, y_pred, marker='.', label='Raw Prediction (Mean MAE: %.3f)' % (mean(scores)))
plt.scatter(y_test, y_test - y_norm_error, marker='.', color='red', label=r'Normalized Prediction $\left(\frac {e_i} {U_{slip, max}}\right)$')
line = np.linspace(np.min(y_test), np.max(y_test))
plt.plot(line, line, color='gray', alpha=0.5, label='True')
plt.xlabel(r'$U_{slip, max}$  $\left(\frac {\mu m}{s}\right)$')
plt.ylabel(r'$U_{slip, max}$  $\left(\frac {\mu m}{s}\right)$')
plt.title(r'$ElasticNet$ Prediction Accuracy on Squires Dataset')
plt.legend(fontsize=12)
plt.show()

# end notes
print("Successful completion")



"""
# define the Elastic Net model
model_orig = ElasticNet(alpha=1.0, l1_ratio=0.5)

sklearn's Elastic Net - a penalized linear regression model with L1 and L2 penalties.
alpha hyperparameter is set via 'l1_ratio' that controls contribution of L1 and L2 penalties.
lambda hyperparameter is set via 'alpha' that controls contribution of the sum of both 
penalties to the loss function.
"""

"""
# fit model to train set
model_orig.fit(X_train, y_train)

# make predictions on test set
yhat = model_orig.score(X_test, y_test)
print('Coefficient of determination (R^2): %.3f' % (yhat))
print('R^2 best possible score = 1')
print('R^2 = 1 - residual sum of squares / total sum of squares')

# ----- manual hyperparameter tuning -----
model_tuned = ElasticNet()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) # repeated 10-fold cross-validation
# define grid search parameter space
grid = dict()
grid['alpha'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
grid['l1_ratio'] = np.arange(0.01, 1, 0.01)
# define search
search = GridSearchCV(model_tuned, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
"""

"""
# ----- tune hyperparameters using built-in ElasticNetCV
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) # repeated 10-fold cross-validation
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
ratios = np.arange(0.01, 1, 0.01)
model_encv_tuned = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
model_encv_tuned.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model_encv_tuned.alpha_)
print('l1_ratio_: %f' % model_encv_tuned.l1_ratio_)
"""

"""
# ----- train and test Elastic Net using tuned hyperparameters -----
# define the Elastic Net model
model_orig = ElasticNet(alpha=1.0, l1_ratio=0.5)
model_orig.fit(X_train, y_train)
yhat = model_orig.score(X_test, y_test)
print('Coefficient of determination (R^2): %.3f' % (yhat))
print('R^2 best possible score = 1')
print('R^2 = 1 - residual sum of squares / total sum of squares')
"""