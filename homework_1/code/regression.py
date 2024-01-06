import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# import the dataset
dataset = pd.read_csv('train_set.tsv', sep='\t', header=0)

# prepare the dataset to be split
dataframe = dataset.drop('num_collisions', axis = 1).copy()

# normalize dataset
scaler = preprocessing.MinMaxScaler()
cols = dataframe.columns
d = scaler.fit_transform(dataframe)
normalized_dataset = pd.DataFrame(d, columns=cols)


X_all = normalized_dataset.drop('min_CPA', axis = 1).copy() # drop the column 'min_CPA' by the dataset
y_all = normalized_dataset['min_CPA'].copy() # create a frame with only the column 'min_CPA' and y represents the thing we want to predict


X_all_res = X_all
y_all_res = y_all


# split the dataset in a training set and in a test set
X_train, X_test, y_train, y_test = train_test_split(X_all_res, y_all_res, test_size=0.33)


# RANDOM FOREST

# set the algorithm
rf_model = RandomForestRegressor(bootstrap=True, max_depth=80, max_features=1, min_samples_leaf=3, min_samples_split=2, n_estimators=100)
# rf_model = RandomForestRegressor()

# train the algorithm
rf_model.fit(X_train, y_train)

# test the algorithm
y_pred_rf = rf_model.predict(X_test)

# grid search for random_forest
# search_space = {
#     'bootstrap': [True],
#     'max_depth': [80, 100],
#     'max_features': [1, 2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [2, 4, 6],
#     'n_estimators': [100, 200]
# }
# gs = GridSearchCV(rf_model, search_space, cv=5)
# gs.fit(X_train, y_train)
# print(gs.best_params_)
# print(gs.best_score_)

# SVR

#set the algorithm
svr_model = SVR(kernel='poly', C=0.2, degree=4)
# svr_model = SVR()

# train the algorithm
svr_model.fit(X_train, y_train)

# test the algorithm
y_pred_svr = svr_model.predict(X_test)

# grid search for svr
# search_space = {
#     'kernel': ['poly'],
#     'C': [0.2, 0.5, 1],
#     'degree': [2, 3, 4]
# }
# gs = GridSearchCV(svr_model, search_space, cv=5)
# gs.fit(X_train, y_train)
# print(gs.best_params_)
# print(gs.best_score_)


# EVALUATION

# mse
svr_mse = mean_squared_error(y_test, y_pred_svr)
print('svr mse: ', svr_mse)

rf_mse = mean_squared_error(y_test, y_pred_rf)
print('rf mse: ', rf_mse)


# re-score
svr_r2 = r2_score(y_test, y_pred_svr)
print('svr r2-score: ', svr_r2)

rf_r2 = r2_score(y_test, y_pred_rf)
print('rf r2-score: ', rf_r2)

# # plot outputs lr
# plt.scatter(X_test.iloc[:,0], y_test,  color='black')
# plt.scatter(X_test.iloc[:,0], y_pred_lr, color='red', linewidth=3)

# plt.xticks(())
# plt.yticks(())
# plt.show()

# # plot outputs svr
# plt.scatter(X_test.iloc[:,0], y_test,  color='black')
# plt.scatter(X_test.iloc[:,0], y_pred_svr, color='red', linewidth=3)

# plt.xticks(())
# plt.yticks(())
# plt.show()
