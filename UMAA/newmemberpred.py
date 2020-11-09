import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report,auc, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


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
    scaled_features = df[columns_to_scale]
    ss = StandardScaler()
    scaled_data = ss.fit_transform(scaled_features)
    df[columns_to_scale] = scaled_data
    df['target'] = df['target'].astype('category')

    return df

def feature_selection(X):
    pca = PCA()
    x_pca = pca.fit_transform(X)
    x_pca = x_pca[:, :12]

    return x_pca

def imbalance(X, y):
    sm = SMOTE(random_state = 42, sampling_strategy= 0.50)
    X, y = sm.fit_sample(X, y.ravel())
    return X, y

def model_building(model_name, X, y):

    if model_name == 'knn':
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 10)}
        knn_cv = GridSearchCV(knn, param_grid, scoring='roc_auc_score', n_jobs=31, cv=5)
        knn_cv.fit(X, y)
        print("best_parameters", knn_cv.best_params_)
        print("best_score", knn_cv.best_score_)
        return knn_cv.best_estimator_

    if model_name == 'rf':
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': np.arange(100, 1000, 100), 'max_depth': np.arange(3, 8, 1), 'max_features':['auto', 'sqrt', 'log2']}
        rf_cv = GridSearchCV(rf, param_grid, scoring='roc_auc_score', n_jobs=31, cv=5)
        rf_cv.fit(X, y)
        print("best_parameters", rf_cv.best_params_)
        print("best_score", rf_cv.best_score_)
        return rf_cv.best_estimator_

    if model_name == 'LGBM':
        param_grid = {'n_estimators':[500], 'learning_rate':[0.01], 'max_depth':[3,4,5]}
        lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)
        lgbm_cv = GridSearchCV(lgbm, param_grid, cv=5, scoring='roc_auc_score')
        lgbm_cv.fit(X, y)
        print("best_parameters", lgbm_cv.best_params_)
        print("best_score", lgbm_cv.best_score_)
        return lgbm_cv.best_estimator_

 if __name__ == "__main__":
     model_list = ['knn', 'rf', 'xgboost', 'lightgbm']
     preprocessing_ind = True
     feature_selection_ind = True
     imbalance_ind = True
     data = pd.read_csv("data_for_model.csv")
     if preprocessing:
         data = preprocessing(data)
     y = data['target']
     X = data.drop(['target'], axis=1)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
     if feature_selection_ind:
         X_train = feature_selection(X_train)
     if imbalance_ind:
         X_train, y_train = imbalance(X_train, y_train)
     est = model_building(model_name = 'LGBM', X = X_train, y = y_train)
     est_1 = model_building(model_name = 'knn', X = X_train, y = y_train)
     est_2 = model_building(model_name = 'rf',  X = X_train, y = y_train)
