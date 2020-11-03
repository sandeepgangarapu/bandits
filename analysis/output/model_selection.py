import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score, \
    f1_score, confusion_matrix, mean_absolute_error, make_scorer
from sklearn.utils import resample
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import metrics
from sklearn.preprocessing import StandardScaler


def import_data(filename):
    """ Function to import csv"""
    data = pd.read_csv(filename, index_col=False)
    return data


def explore_data(data):
    """ Has many commands to help understand the data better"""
    print data.head()
    print data.info()
    print data.shape
    print data.describe()
    print data.apply(pd.Series.value_counts)


def clean_data(data, all_cols, remove_na=True):
    """ Function to perform cleaning operations"""
    if remove_na:
        data = data.dropna(axis=0, how='any')
    data = data[all_cols]
    return data


def convert_to_categorical(data, colnames):
    """ Converts some data values into categorical"""
    for col in colnames:
        data[col] = data[col].astype('category')
    return data

def remove_imbalance(data, outcome):
    data_majority = data[data[outcome] == 0]
    data_minority = data[data[outcome] == 1]
    print data[outcome].value_counts()

    # Upsample minority class
    data_minority_upsampled = resample(data_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=56973,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    data_balance = pd.concat([data_majority, data_minority_upsampled])

    # Display new class counts
    print data_balance[outcome].value_counts()
    return data_balance


def performance_metrics(test_array, pred_array, metrics):
    if "accuracy_score" in metrics:
        print "accuracy_score"
        print accuracy_score(test_array, pred_array)

    if "cohen_kappa_score" in metrics:
        print "cohen_kappa_score"
        print cohen_kappa_score(test_array, pred_array)

    if "roc_auc_score" in metrics:
        print "roc_auc_score"
        print roc_auc_score(test_array, pred_array)

    if "f1_score" in metrics:
        print "f1_score"
        print f1_score(test_array, pred_array)

    if "confusion_matrix" in metrics:
        print "confusion_matrix"
        print confusion_matrix(test_array, pred_array)

def random_forest_classification(x, y, metric):
    print "****************************"
    print "random_forest_classification"
    print "****************************"

    rfr = RandomForestClassifier(n_estimators=100)
    param_grid = {'max_depth': np.append(np.arange(2,10,2),np.arange(15,31,5)), 'min_samples_leaf': np.arange(1,21,4),
                  'max_features': ['auto','sqrt','log2']}
    rfr_cv = GridSearchCV(rfr, param_grid, cv=5, scoring=make_scorer(metric, greater_is_better=True), n_jobs=30)
    rfr_cv.fit(x, y)
    bscore = rfr_cv.best_score_
    bparams = rfr_cv.best_params_
    return rfr_cv

def svc_param_selection(x, y, metric):
    print "****************************"
    print "svc_param_selection"
    print "****************************"

    svm_model = svm.SVC(kernel='rbf')
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring=make_scorer(metric, greater_is_better=True), n_jobs=30)
    grid_search.fit(x, y)
    return grid_search

if __name__ == "__main__":

    input_filename = "donor_general_input.csv"
    all_cols = ['weight_cat', 'age_cat', 'male', 'married', 'resident_cat', 'edu_cat',
                  'occupation_cat', 'bdonationvol', 'recency_in_month', 'formvoluntary',
                  'formblood', 'formplasma', 'formmutual', 'formgroup', 'target']
    covariates = ['weight_cat', 'age_cat', 'male', 'married', 'resident_cat', 'edu_cat',
                  'occupation_cat', 'bdonationvol', 'recency_in_month', 'formvoluntary',
                  'formblood', 'formplasma', 'formmutual', 'formgroup']
    categorical_colnames = ['weight_cat', 'age_cat', 'male', 'married', 'resident_cat', 'edu_cat',
                            'occupation_cat','target']
    outcome_variable = 'target'
    # Method to import data
    data = import_data(filename=input_filename)

    # Method to get to know data
    # explore_data(data = data)

    # Method to clean data
    data = clean_data(data, all_cols, remove_na=True)

    # Convert some columns to categorical
    data = convert_to_categorical(data, colnames=categorical_colnames)
    #
    # data_balance = remove_imbalance(data, outcome = outcome_variable)
    # # Method to split data into test and train
    # print "--------------------------"
    # print "Imbalance Data"
    # print "--------------------------"
    train_x, test_x, train_y, test_y_imb = train_test_split(data[covariates], data[outcome_variable],
                                                                      test_size=0.30,
                                                                      stratify=data[outcome_variable],
                                                                      random_state=1)

    # rf_model = random_forest_classification(train_x, train_y, metric=f1_score)
    # y_predict = rf_model.predict(test_x)
    # performance_metrics(test_y_imb, y_predict, metrics=["f1_score"])
    #
    # svm_model = svc_param_selection(train_x, train_y, metric=f1_score)
    # y_predict = svm_model.predict(test_x)
    # performance_metrics(test_y_imb, y_predict, metrics=["f1_score"])
    #
    # print "--------------------------"
    # print "Balance Data"
    # print "--------------------------"
    # train_x, test_x, train_y, test_y_bal = train_test_split(data_balance[covariates], data_balance[outcome_variable],
    #                                                     test_size=0.30,
    #                                                     stratify=data_balance[outcome_variable],
    #                                                     random_state=1)
    #
    #
    # rf_model = random_forest_classification(train_x, train_y, metric=f1_score)
    # y_predict = rf_model.predict(test_x)
    # performance_metrics(test_y_bal, y_predict, metrics=["f1_score"])
    #
    # svm_model = svc_param_selection(train_x, train_y, metric=f1_score)
    # y_predict = rf_model.predict(test_x)
    # performance_metrics(test_y_bal, y_predict, metrics=["f1_score"])
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    num_classes = 2
    train_y = np_utils.to_categorical(train_y, num_classes)
    test_y_imb = np_utils.to_categorical(test_y_imb, num_classes)

    model = Sequential()

    # Add an input layer
    model.add(Dense(12, activation='relu', input_shape=(14,)))

    # Add one hidden layer
    model.add(Dense(8, activation='relu'))

    # Add an output layer
    model.add(Dense(2, activation='sigmoid'))

    # Model output shape
    model.output_shape

    # Model summary
    model.summary()

    # Model config
    model.get_config()

    # List all weight tensors
    model.get_weights()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.categorical_accuracy])

    model.fit(train_x, train_y, epochs=20, batch_size=1, verbose=1)

    y_pred = model.predict(test_x)

    pred_y = [0 if x<0.5 else 1 for x in y_pred]

    performance_metrics(test_y_imb, y_pred, ["f1_score","confusion_matrix"])

