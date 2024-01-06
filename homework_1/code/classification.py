import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# import the dataset
dataset = pd.read_csv('train_set.tsv', sep='\t', header=0)

# prepare the dataset to be split
X_all_temp = dataset.drop('min_CPA', axis = 1).copy() # drop the column 'min_CPA' by the dataset
X_all = X_all_temp.drop('num_collisions', axis = 1).copy() # drop the column 'num_collisions' by the dataset and X represents the columns to make classification
y_all = dataset['num_collisions'].copy() # create a frame with only the column 'num_collisions' and y represents the thing we want to predict

# balance the dataset
# ros = RandomOverSampler()
# X_all_res, y_all_res = ros.fit_resample(X_all, y_all)

# print(y_all_res.value_counts())

X_all_res = X_all
y_all_res = y_all

# normalize features
scaler = preprocessing.MinMaxScaler()
cols = X_all_res.columns
d = scaler.fit_transform(X_all_res)
normalized_X_all_res = pd.DataFrame(d, columns=cols)

# split the dataset in a training set and in a test set
X_train, X_test, y_train, y_test = train_test_split(normalized_X_all_res, y_all_res, test_size=0.33)


# balance the training set
ros = RandomOverSampler()
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
y_train_res.value_counts().plot.pie(autopct='%.2f')

X_train = X_train_res
y_train = y_train_res


# SVM

# set the algorithm
svm_model = svm.SVC(C=1, gamma=1, kernel='rbf')
# svm_model = svm.SVC(class_weight={0:1, 1:int(Counter(y_all)[0]/Counter(y_all)[1]), 2:int(Counter(y_all)[0]/Counter(y_all)[2]), 3:int(Counter(y_all)[0]/Counter(y_all)[3]), 4:int(Counter(y_all)[0]/Counter(y_all)[4])})
# svm_model = svm.SVC()


# train the algorithm
svm_model.fit(X_train, y_train)

# test the algorithm
y_pred_svm = svm_model.predict(X_test)

# grid search for svm
# search_space = {
#     'C': [0.2, 0.5, 1], 
#     'gamma': [1],
#     'kernel': ['rbf', 'linear']
# }
# gs = GridSearchCV(svm_model, search_space, cv=5)
# gs.fit(X_train, y_train)
# print(gs.best_params_)
# print(gs.best_score_)


# RF

# set the model
rf_model = RandomForestClassifier(bootstrap=True, max_depth=100, max_features=2, min_samples_leaf=3, min_samples_split=8, n_estimators=300)
# rf_model = RandomForestClassifier()
# rf_model = RandomForestClassifier(class_weight={0:1, 1:int(Counter(y_all)[0]/Counter(y_all)[1]), 2:int(Counter(y_all)[0]/Counter(y_all)[2]), 3:int(Counter(y_all)[0]/Counter(y_all)[3]), 4:int(Counter(y_all)[0]/Counter(y_all)[4])})

# train the model
rf_model.fit(X_train, y_train)

# test the algorithm
y_pred_rf = rf_model.predict(X_test)

# grid search for random_forest
# search_space = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300]
# }
# gs = GridSearchCV(rf_model, search_space, cv=5)
# gs.fit(X_train, y_train)
# print(gs.best_params_)
# print(gs.best_score_)




# EVALUATION

# accuracy
svm_acc = svm_model.score(X_test, y_test)
print("\nSVM accuracy:", svm_acc)

rf_acc = rf_model.score(X_test, y_test)
print("RF accuracy", rf_acc)

# precision, recall, f1-score
print("\nSVM table")
print(classification_report(y_test, y_pred_svm, labels=None, digits=3))

print("\nRF table")
print(classification_report(y_test, y_pred_rf, labels=None, digits=3))

# confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_) # this is the confusion matrix
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=svm_model.classes_)
disp_svm.plot()
plt.show() # this print the confusion matrix

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_) # this is the confusion matrix
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)
disp_rf.plot()
plt.show() # this print the confusion matrix

#k-cross validation
# print("\nSVM cross-validation")
# cv_svm = ShuffleSplit(n_splits=5, test_size=0.2)
# scores_svm = cross_val_score(svm_model, X_all_res, y_all_res, cv=cv_svm)
# print(scores_svm)
# print("Accuracy: %0.3f (standard deviation: +/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

# print("\nRF cross-validation")
# cv_rf = ShuffleSplit(n_splits=5, test_size=0.2)
# scores_rf = cross_val_score(rf_model, X_all_res, y_all_res, cv=cv_rf)
# print(scores_rf)
# print("Accuracy: %0.3f (standard deviation: +/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

# roc curve
# svm_fpr, svm_tpr, _ = roc_curve(y_test, y_pred_svm, pos_label=1)
# svm_auc = auc(svm_fpr, svm_tpr)

# rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_rf, pos_label=1)
# rf_auc = auc(rf_fpr, rf_tpr)

# plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % svm_auc)
# plt.plot(rf_fpr, rf_tpr, marker='.', label='RF (auc = %0.3f)' % rf_auc)

# plt.xlabel('False Positive Rate -->')
# plt.ylabel('True Positive Rate -->')

# plt.legend()

# plt.show()
