import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import tpot

def preprocessing(df):
    """# Preprocessing
    * Missing value treatment
    * Change to appropriate data types
    * Normalization
    """
    #  filling null values with popular occurance of that **variable**
    df = df.fillna(data['IS_CURRENT_TC_EMPLOYEE'].value_counts().index[0])
    # convert the binary columns into categorical variable
    col_names_object = ['GENDER', 'IN_TC_METRO_AREA', 'IS_CURRENT_TC_EMPLOYEE', 'ATHLETIC_INTEREST', 'TRAVEL_INTEREST', 'AFFINITY_NETWORK_INTEREST', 'WEB_TOPIC_OPT_INS']

    for col in col_names_object:
      df[col] = df[col].astype('category')

    # removing this id as its not a feature
    df = df.drop(["ID_DEMO"], axis=1)

    # one hot encoding of marital status
    df = pd.get_dummies(df, columns=['MARITAL_STATUS'], prefix = ['MARITAL_STATUS'])
    df = df.drop(['MARITAL_STATUS_O'], axis = 1)
    df['MARITAL_STATUS_U'] = df['MARITAL_STATUS_U'].astype('category')
    df['MARITAL_STATUS_M'] = df['MARITAL_STATUS_M'].astype('category')
    # df['target'] = df['target'].astype('category')
    return df


def Normalization(X_train, X_test):
    # Normalization
    # except age everything has low variance, no need for log normalization
    # but we can do scalar tranformation on the rest of the columns
    columns_to_scale = ['AGE','UMN_member', 'UMN_donor',
           'UMN_volun', 'UMN_inform', 'UMN_loyalty',
           'UMN_avg_Annual_score_5_years', 'annual_member', 'life_member',
           'non_member', 'Learning_emails', 'Legislature_emails',
           'Mass Updates_emails', 'Other_emails', 'Social_emails',
           'Solicitations_emails', 'Sports_emails', 'general_ctr_emails',
           'Learning_events', 'Legislature_events', 'Networking_events',
           'Other_events', 'Social_events', 'Sports_events',
           'total_type_person_events']
    scaled_features_train = X_train[columns_to_scale]
    ss = StandardScaler()
    scaled_data_train = ss.fit_transform(scaled_features_train)
    X_train[columns_to_scale] = scaled_data_train
    scaled_features_test = X_test[columns_to_scale]
    scaled_data_test = ss.transform(scaled_features_test)
    X_test[columns_to_scale] = scaled_data_test
    return X_train, X_test


def feature_selection(X_train, X_test):
    pca = PCA()
    x_pca_train = pca.fit_transform(X_train)
    x_pca_train = x_pca_train[:, :12]

    x_pca_test = pca.transform(X_test)
    x_pca_test = x_pca_test[:, :12]

    return x_pca_train, x_pca_test

def imbalance(X, y):
    sm = SMOTE(random_state=42, sampling_strategy=1)
    X, y = sm.fit_sample(X, y.ravel())
    # print(np.unique(y, return_counts=True))
    return X, y

def model_building(model_name, X, y):

    if model_name == 'knn':
        knn = KNeighborsClassifier(weights='distance')
        param_grid = {'n_neighbors': np.arange(3, 10)}
        knn_cv = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=5)
        print(X.dtype, y.dtype)
        knn_cv.fit(X, y)
        print("best_parameters", knn_cv.best_params_)
        print("best_score", knn_cv.best_score_)
        print("scoring", knn_cv.scorer_)
        print("prob", knn_cv.predict_proba)
        return knn_cv.best_estimator_

    if model_name == 'rf':
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': np.arange(100, 1000, 100), 'max_depth': np.arange(3, 8, 1), 'max_features':['auto', 'sqrt', 'log2']}
        rf_cv = GridSearchCV(rf, param_grid, scoring='roc_auc', cv=5)
        rf_cv.fit(X, y)
        print("best_parameters", rf_cv.best_params_)
        print("best_score", rf_cv.best_score_)
        print("scoring", rf_cv.scorer_)
        print("prob", rf_cv.predict_proba)
        return rf_cv.best_estimator_

    if model_name == 'LGBM':
        param_grid = {
            'n_estimators': [400, 700, 1000],
            'colsample_bytree': [0.7, 0.8],
            'max_depth': [15,20,25],
            'num_leaves': [50, 100, 200]
        }
        lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)
        lgbm_cv = GridSearchCV(lgbm, param_grid, n_jobs=-1, cv=5, scoring='roc_auc')
        lgbm_cv.fit(X, y)
        print("best_parameters", lgbm_cv.best_params_)
        print("best_score", lgbm_cv.best_score_)
        return lgbm_cv.best_estimator_

    if model_name == 'tpot':
        pipeline_optimizer = tpot.TPOTClassifier(cv=5, n_jobs=-1, scoring='roc_auc', verbosity=2)
        pipeline_optimizer.fit(X_train, y_train)  # fit the pipeline optimizer - can take a long time
        print(pipeline_optimizer.score(X_test, y_test))  # print scoring for the pipeline
        pipeline_optimizer.export('tpot_exported_pipeline.py')  # export the pipeline - in Python code!
        return None

if __name__ == "__main__":
     model_list = ['knn', 'rf', 'xgboost', 'lightgbm']
     preprocessing_ind = True
     normalization = True
     feature_selection_ind = True
     imbalance_ind = True
     data = pd.read_csv("data_for_model.csv")
     if preprocessing:
         data = preprocessing(data)
     y = data.target.ravel()
     X = data.drop(['target'], axis=1)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
     print(np.unique(y_train, return_counts=True))
     print(np.unique(y_test, return_counts=True))
     if normalization:
         X_train, X_test = Normalization(X_train, X_test)
     if feature_selection_ind:
         X_train, X_test = feature_selection(X_train, X_test)
     if imbalance_ind:
         X_train, y_train = imbalance(X_train, y_train)
     est = model_building(model_name = 'LGBM', X = X_train, y = y_train)
     #est_1 = model_building(model_name = 'knn', X=X_train, y=y_train)
     est_2 = model_building(model_name='tpot', X=X_train, y=y_train)
