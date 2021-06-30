# -*- coding: utf-8 -*-
"""Thesis_Submition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MzhQR0x986eUr_FT_LPhb3G6tvmPmmPF

## A. Mounting Drive into Google Colab
"""

from google.colab import drive
drive.mount('/content/drive')

"""## B. Installing dependencies

Before running the present script make sure you install the following library dependencies in the order stated:

1. !pip install catboost
2. !pip uninstall lightgbm -y
3. ! git clone --recursive https://github.com/Microsoft/LightGBM
4. ! cd LightGBM && rm -rf build && mkdir build && cd build && cmake -DUSE_GPU=1 ../../LightGBM && make -j4 && cd ../python-package && python3 setup.py install --precompile --gpu;
"""

#1. Installing CATBOOST
!pip install catboost

#2. Installing LIGHTBOOST
!pip uninstall lightgbm -y

# Cloning LGBM git repository
! git clone --recursive https://github.com/Microsoft/LightGBM

#4. Setting up GPU for LGBM
! cd LightGBM && rm -rf build && mkdir build && cd build && cmake -DUSE_GPU=1 ../../LightGBM && make -j4 && cd ../python-package && python3 setup.py install --precompile --gpu;

"""## C. Importing custom functionalitites"""

# Custom functionalities path
import sys
sys.path.append('/content/drive/MyDrive/')

# Impoting standard dependencies
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Impoting customized functionalities
import module_lithopred
from module_lithopred.data_formating import formating
from module_lithopred import plotting
from module_lithopred.plotting import raw_logs, augmented_logs, litho_prediction
from module_lithopred.preprocessing import preprocess_data
from module_lithopred.augmentation import data_augmentation
from module_lithopred.input_norm import normalization
from module_lithopred.additional_functions import matrix_score, confusion_matrix

"""## D. Importing data"""

# Setting datasets directories
directory = '/content/drive/MyDrive/Thesis_data/'

"""Imporitng training, open test, and hidden test sets
"""
raw_training = pd.read_csv(directory + 'train.csv', sep=';') # rawtraining dataset

test_data = pd.read_csv(directory + 'test.csv', sep=';')
test_labels = pd.read_csv(directory + 'test_target.csv', sep=';')
raw_test = pd.merge(test_data, test_labels, on=['WELL', 'DEPTH_MD']) #open test dataset

raw_hidden = pd.read_csv(directory + 'hidden_test.csv', sep=';')  #hidden test dataset

#Data formating

""" Calling formating function, which renames columns and maps lithofacies
classes values from 0 to 11, and drops intepretation confidence column.
"""

training_form, test_form, hidden_form = formating(raw_training, raw_test, raw_hidden)

#Inspecting raw data
display(training_form.head())

""" Formated well logs visualization by calling raw_logs 
customized function. Only the fist well displayed.

For plotting additionall wells change the range(0, 1).

See raw_logs.py for further details about the function.
"""
for i in range(0, 1):
  raw_logs(training_form, i)

"""## E. Data Pre-processing"""

""" Preprocessing involves dropping unncesary cols, encoding categorical variables,
and incorporating well location clustering as feature prior to feature augmentation.

See prerpocessing.py for further deatils.
"""

traindata, testdata, hiddendata = preprocess_data(training_form, test_form, hidden_form)

#Checking data structure befrore augmentation
display(traindata.head())

"""## F. Feature augmentation by Machine-learning"""

"""First, each regressor takes as training data the 80% of the features
where the log being predicted is present, later each regresor is validated
on the remanent 20% of this data, and tested on the open test set.

Second, each regression model predictcs and imputes the predicted value 
where the predicted log readings were originally missing on the training,
open, and hidden sets.

Finally, some other additional features are created.

See argumentation.py for further details.
"""

training_aug, test_aug, hidden_aug = data_augmentation(traindata, testdata, hiddendata)

# Inspecting dataframe after augmentation

display(training_aug.head())

"""Concatenating the actual, predited, and augmented well logs for plotting
"""

cols_needed = ['DTS', 'DTS_pred', 'DTS_COMB', 'NPHI', 'NPHI_pred', 'NPHI_COMB', 'RHOB',
               'RHOB_pred', 'RHOB_COMB', 'DTC', 'DTC_pred', 'DTC_COMB', 'LITHO']

train_predicted_logs = pd.concat((raw_training[['WELL', 'DEPTH_MD']],
                                  training_aug[cols_needed].reset_index()), axis=1)

test_predicted_logs = pd.concat((raw_test[['WELL', 'DEPTH_MD']],
                                 test_aug[cols_needed].reset_index()), axis=1)

hidden_predicted_logs = pd.concat((raw_hidden[['WELL', 'DEPTH_MD']],
                                   hidden_aug[cols_needed].reset_index()), axis=1)

""" Actual, predicted, and augmented well logs visualization by calling 
augmented_logs customized function. Only the fist well displayed.

For plotting additionall wells change the range(0, 1).

See augmented_logs.py for further details about the function.
"""

for i in range(10, 11):
  augmented_logs(train_predicted_logs, i)

"""## G. Data Normalization"""

"""First, the features that were not augmented by machine-learning
are inputed by median inputation technique before normalizing

Later, a Standard Scaler is used to standardize the datasets.

See normalization.py for further details.
"""

train_norm, test_norm, hidden_norm = normalization(training_aug, test_aug, hidden_aug)

"""## H. Machine learning models' results

#### A1. LOGISTIC REGRESSION MODEL (LR)
"""

"""Predicting lithofacies by using the Logistic Regression model
(LR) by calling run_LR function.

See LR_model.py for further details about the function.
"""

from module_lithopred.ML_models.LR_model import run_LR
train_pred_lr, test_pred_lr, hidden_pred_lr = run_LR(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_lr))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_lr))

"""Storing LR predctions into a copy of the formated datasets.
"""
train_lr_res = training_form.copy()
test_lr_res = test_form.copy()
hidden_lr_res = hidden_form.copy()

#Appending predictions to a copy of the augmented datasets
train_lr_res['LR_TM'] = train_pred_lr
test_lr_res['LR_TM'] = test_pred_lr
hidden_lr_res['LR_TM'] = hidden_pred_lr

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_lr_res, i, 1)

"""#### A2. K-NEAREST NEIGHBOR MODEL (KNN)"""

"""Predicting lithofacies by using the K-Nearest Neighbor model
(KNN) by calling run_KNN function.

See KNN_model.py for further details about the function.
"""

from module_lithopred.ML_models.KNN_model import run_KNN
train_pred_knn, test_pred_knn, hidden_pred_knn = run_KNN(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_knn))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_knn))

"""Storing KNN predctions into a copy of the formated datasets.
"""
train_knn_res = training_form.copy()
test_knn_res = test_form.copy()
hidden_knn_res = hidden_form.copy()

#Appending predictions to a copy of the augmented datasets
train_knn_res['KNN_TM'] = train_pred_knn
test_knn_res['KNN_TM'] = test_pred_knn
hidden_knn_res['KNN_TM'] = hidden_pred_knn

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_knn_res, i, 1)

"""#### A3. SUPPORT VECTOR MACHINES"""

"""Predicting lithofacies by using the Support Vector Machines model
(SVM) by calling run_SVM function.

See SVM_model.py for further details about the function.
"""

from module_lithopred.ML_models.SVM_model import run_SVM
train_pred_svm, test_pred_svm, hidden_pred_svm = run_SVM(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""
 
print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_svm))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_svm))

"""Storing SVM predctions into a copy of the formated datasets.
"""
train_svm_res = training_form.copy()
test_svm_res = test_form.copy()
hidden_svm_res = hidden_form.copy()

train_svm_res['SVM_TM'] = train_pred_svm
test_svm_res['SVM_TM'] = test_pred_svm
hidden_svm_res['SVM_TM'] = hidden_pred_svm

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_svm_res, i, 1)

"""#### A4. DECISION TREES"""

"""Predicting lithofacies by using the Decision Tree model
(DT) by calling run_DT function.

See DT_model.py for further details about the function.
"""

from module_lithopred.ML_models.DT_model import run_DT
train_pred_dt, open_pred_dt, hidden_pred_dt = run_DT(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_dt))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_dt))

"""Storing DT predctions into a copy of the formated datasets.
"""
train_dt_res = training_form.copy()
test_dt_res = test_form.copy()
hidden_dt_res = hidden_form.copy()

train_dt_res['DT_TM'] = train_pred_dt
test_dt_res['DT_TM'] = open_pred_dt
hidden_dt_res['DT_TM'] = hidden_pred_dt

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_dt_res, i, 1)

"""#### A5. RANDOM FOREST"""

"""Predicting lithofacies by using the Random Forest model
(RF) by calling run_RF function.

See RF_model.py for further details about the function.
"""

from module_lithopred.ML_models.RF_model import run_RF
train_pred_rf, open_pred_rf, hidden_pred_rf = run_RF(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""
 
print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_rf))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_rf))

"""Storing RF predctions into a copy of the formated datasets.
"""
train_rf_res = training_form.copy()
test_rf_res = test_form.copy()
hidden_rf_res = hidden_form.copy()

#Appending predictions to a copy of the augmented datasets
train_rf_res['RF_TM'] = train_pred_rf
test_rf_res['RF_TM'] = open_pred_rf
hidden_rf_res['RF_TM'] = hidden_pred_rf

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_rf_res, i, 1)

"""#### A6. CATEGORICAL GRADIENT BOOSTING MODEL (CATBOOST)"""

"""Predicting lithofacies by using the Categorical Gradient Boosting model
(CATBOST) by calling run_CatBoost function.

See CatBoost_model.py for further details about the function.
"""
from module_lithopred.ML_models.CatBoost_model import run_CatBoost

train_pred_cat, test_pred_cat, hidden_pred_cat = run_CatBoost(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_cat))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_cat))

"""Storing CATBOOST predctions into a copy of the formated datasets.
"""
train_cat_res = training_form.copy()
test_cat_res = test_form.copy()
hidden_cat_res = hidden_form.copy()

#Appending predictions to a copy of the augmented datasets
train_cat_res['CAT_TM'] = train_pred_cat
test_cat_res['CAT_TM'] = test_pred_cat
hidden_cat_res['CAT_TM'] = hidden_pred_cat

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_cat_res, i, 1)

"""#### A7. LIGTH GRADIENT BOOSTING MODEL (LIGTHBOOST)"""

"""Predicting lithofacies by using the Light Gradient Boosting model
(LGBM) by calling run_CatBoost function.

See LightBoost_model.py for further details about the function.
"""
from module_lithopred.ML_models.LightBoost_model import run_LightBoost
train_pred_light, test_pred_light, hidden_pred_light = run_LightBoost(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_light))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_light))

"""Storing LGBM predctions into a copy of the formated datasets.
"""
train_light_res = training_form.copy()
test_light_res = test_form.copy()
hidden_light_res = hidden_form.copy()

train_light_res['LIGHT_TM'] = train_pred_light
test_light_res['LIGHT_TM'] = test_pred_light
hidden_light_res['LIGHT_TM'] = hidden_pred_light

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_light_res, i, 1)

"""#### A8. EXTREME GRADIENT BOOSTING MODEL (XGB)"""

"""Predicting lithofacies by using the eXtreme Gradient Boosting model (XGB)
by calling run_XGB function.

See XGB_model.py for further details about the function.
"""
from module_lithopred.ML_models.XGB_model import run_XGB

train_pred_xgb, test_pred_xgb, hidden_pred_xgb = run_XGB(train_norm, test_norm, hidden_norm)

""""Plotting the classification report for the hidden test set
and the matrix penalty score.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_xgb))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_xgb))

"""Storing XGB predctions into a copy of the formated datasets.
"""
train_xgb_res = training_form.copy()
test_xgb_res = test_form.copy()
hidden_xgb_res = hidden_form.copy()

train_xgb_res['XGB_TM'] = train_pred_xgb
test_xgb_res['XGB_TM'] = test_pred_xgb
hidden_xgb_res['XGB_TM'] = hidden_pred_xgb

""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 1):
  litho_prediction(hidden_xgb_res, i, 1)

"""#### A9.NEURAL NETWORK MODEL (NN)"""

"""Predicting lithofacies by using the Neural Network
(NN) by calling run_NN function.

See NN_model.py for further details about the function.
"""

from module_lithopred.ML_models.NN_model import run_NN
train_pred_nn, test_pred_nn, hidden_pred_nn = run_NN(train_norm, test_norm, hidden_norm)

#Printing Classification Reports
""""Plotting the classification report for the hidden test set
and the matrix penalty score.

See additional_functions.py for further details about  matrix_score
and classification_report functions.
"""

print('-----------------------HIDDEN SET REPORT---------------------')
print('Hidden set penalty matrix score:', matrix_score(hidden_norm.LITHO.values, hidden_pred_nn))
print('Hidden set report:', classification_report(hidden_norm.LITHO, hidden_pred_nn))

"""Storing NN predctions into a copy of the formated datasets.
"""
train_nn_res = training_form.copy()
test_nn_res = test_form.copy()
hidden_nn_res = hidden_form.copy()

#Appending predictions to a copy of the augmented datasets
train_nn_res['NN_TM'] = train_pred_nn
test_nn_res['NN_TM'] = test_pred_nn
hidden_nn_res['NN_TM'] = hidden_pred_nn

#Plotting HIDDEN SET results
""" Plotting hidden test set well logs, actual and predicted litholofacies by calling 
litho_prediction customized function. Only the fist well is displayed.

For plotting additionall wells change the range(0, 1).

See litho_prediction.py for further details about the function.
"""

for i in range(0, 10):
  litho_prediction(hidden_nn_res, i, 1)