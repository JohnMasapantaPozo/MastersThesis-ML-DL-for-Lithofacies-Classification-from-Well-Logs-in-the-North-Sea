
def data_augmentation(traindata, testdata, hiddendata):
  import numpy as np
  import pandas as pd
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import max_error, mean_squared_error

  ##1. PREDICTING DTS

  print('-----------------------------------------PREDINCTING DTS------------------------------------------')
  '''
  Getting data containing DTS readings and 
  spliting into a new training and validation subsets.

  '''
  traindata_dts = traindata[traindata.DTS.notna()]
  X_dts = traindata_dts.drop(['LITHO', 'DTS'], axis=1)
  Y_dts = traindata_dts['DTS']

  '''
  Inputing remanent features by their median.

  '''

  X_dts_inp = X_dts.apply(lambda x: x.fillna(x.median()), axis=0)                      #Imputation

  X_dts_train, X_dts_val, Y_dts_train, Y_dts_val = train_test_split(X_dts_inp, Y_dts, 
                                                                    test_size=0.3, 
                                                                    random_state=42)   #train validation split

  print('Training set shape {} and validation set shape {}'.format(X_dts_train.shape, X_dts_val.shape))

  '''
  Runnning XGB Regressor

  '''
  model1000 = XGBRegressor(tree_method='gpu_hist'
                           )
  
  model1000.fit(X_dts_train, Y_dts_train.values.ravel(), early_stopping_rounds=100, eval_set=[(X_dts_val, Y_dts_val)], verbose=100)
  train_pred = model1000.predict(X_dts_train)
  val_pred = model1000.predict(X_dts_val)

  print('Train set root mean squared error:', np.sqrt(mean_squared_error(Y_dts_train, train_pred)))
  print('Validation set root mean squared error:', np.sqrt(mean_squared_error(Y_dts_val, val_pred)))

  print('--------------------------------Imputing DTS LOG by ML predictions--------------------------------')

  '''
  Inputing median in the whole sets 
  before predinting DTS.

  '''
  # Filling nan values before predicting DTS
  X_train_DTS = traindata.drop(['LITHO', 'DTS'], axis=1)
  X_train_DTS2 = X_train_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_DTS = testdata.drop(['LITHO', 'DTS'], axis=1)
  X_test_DTS2 = X_test_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_DTS = hiddendata.drop(['LITHO', 'DTS'], axis=1)
  X_hidden_DTS2 = X_hidden_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  '''
  Predicting DTS and 
  filling empty spaces by predictions.

  '''

  #Predicting DTS (COMPLETE DATASETS)
  traindata['DTS_pred'] = model1000.predict(X_train_DTS2)
  testdata['DTS_pred'] = model1000.predict(X_test_DTS2)
  hiddendata['DTS_pred'] = model1000.predict(X_hidden_DTS2)

  #Inputing nan values in DTS with DTS_PREDICTED
  traindata['DTS_COMB'] = traindata['DTS']
  traindata['DTS_COMB'].fillna(traindata['DTS_pred'], inplace=True)

  testdata['DTS_COMB'] = testdata['DTS']
  testdata['DTS_COMB'].fillna(testdata['DTS_pred'], inplace=True)

  hiddendata['DTS_COMB'] = hiddendata['DTS']
  hiddendata['DTS_COMB'].fillna(hiddendata['DTS_pred'], inplace=True)
  



  ##2. PREDICTING NPHI

  print('----------------------------------------PREDINCTING NPHI-----------------------------------------')
  '''
  Getting data containing NPHI readings and 
  spliting into a new training and validation subsets.

  '''
  traindata_nphi = traindata[traindata.NPHI.notna()]
  X_nphi = traindata_nphi.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  Y_nphi = traindata_nphi['NPHI']

  '''
  Inputing remanent features by their median.

  '''

  X_nphi_inp = X_nphi.apply(lambda x: x.fillna(x.median()), axis=0)                          #Imputation
  X_nphi_train, X_nphi_val, Y_nphi_train, Y_nphi_val = train_test_split(X_nphi_inp, Y_nphi, 
                                                                        test_size=0.3, 
                                                                        random_state=42)     #train-validation split

  print('Training set sahpe {} and validation set shape {}'.format(X_nphi_train.shape, X_nphi_val.shape))

  '''
  Runnning XGB Regressor

  '''
  model2000 = XGBRegressor(tree_method='gpu_hist')
  
  model2000.fit(X_nphi_train, Y_nphi_train.values.ravel(), early_stopping_rounds=100, eval_set=[(X_nphi_val, Y_nphi_val)], verbose=100)
  train_pred = model2000.predict(X_nphi_train)
  val_pred = model2000.predict(X_nphi_val)

  print('Training set root mean squared error:', np.sqrt(mean_squared_error(Y_nphi_train, train_pred)))
  print('Validation set root mean squared error:', np.sqrt(mean_squared_error(Y_nphi_val, val_pred)))

  print('--------------------------------Imputing NPHI LOG by ML predictions--------------------------------')

  '''
  Inputing median in the whole sets 
  before predinting DTS.

  '''

  # Filling nan values before predicting NPHI
  X_train_NPHI = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_train_NPHI2 = X_train_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_NPHI = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_test_NPHI2 = X_test_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_NPHI = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_hidden_NPHI2 = X_hidden_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  '''
  Predicting NPHI and 
  filling empty spaces by predictions.

  '''

  #Predicting DTS (COMPLETE DATASETS)
  traindata['NPHI_pred'] = model2000.predict(X_train_NPHI2)
  testdata['NPHI_pred'] = model2000.predict(X_test_NPHI2)
  hiddendata['NPHI_pred'] = model2000.predict(X_hidden_NPHI2)

  #Inputing nan values in DTS with DTS_PREDICTED
  traindata['NPHI_COMB'] = traindata['NPHI']
  traindata['NPHI_COMB'].fillna(traindata['NPHI_pred'], inplace=True)

  testdata['NPHI_COMB'] = testdata['NPHI']
  testdata['NPHI_COMB'].fillna(testdata['NPHI_pred'], inplace=True)

  hiddendata['NPHI_COMB'] = hiddendata['NPHI']
  hiddendata['NPHI_COMB'].fillna(hiddendata['NPHI_pred'], inplace=True)




  ##3. PREDICTING RHOB
  print('----------------------------------------PREDINCTING RHOB-----------------------------------------')
  '''
  Getting data containing RHOB readings and 
  spliting into a new training and validation subsets.

  '''
  traindata_rhob = traindata[traindata.RHOB.notna()]
  X_rhob = traindata_rhob.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  Y_rhob = traindata_rhob['RHOB']

  '''
  Inputing remanent features by their median.

  '''

  X_rhob_inp = X_rhob.apply(lambda x: x.fillna(x.median()), axis=0)                           #Imputation
  X_rhob_train, X_rhob_val, Y_rhob_train, Y_rhob_val = train_test_split(X_rhob_inp, Y_rhob, 
                                                                        test_size=0.3, 
                                                                        random_state=42)       #train-validation split

  print('Training set sahpe {} and validation set shape {}'.format(X_rhob_train.shape, X_rhob_val.shape))

  '''
  Runnning XGB Regressor

  '''
  model4000 = XGBRegressor(tree_method='gpu_hist')
  
  model4000.fit(X_rhob_train, Y_rhob_train.values.ravel(), early_stopping_rounds=100, eval_set=[(X_rhob_val, Y_rhob_val)], verbose=100)
  train_pred = model4000.predict(X_rhob_train)
  val_pred = model4000.predict(X_rhob_val)

  print('Train set root mean squared error:', np.sqrt(mean_squared_error(Y_rhob_train, train_pred)))
  print('Validation set root mean squared error:', np.sqrt(mean_squared_error(Y_rhob_val, val_pred)))
  
  print('--------------------------------Imputing RHOB LOG by ML predictions--------------------------------')

  '''
  Inputing median in the whole sets 
  before predinting RHOB.

  '''

  # Filling nan values before predicting NPHI
  X_train_RHOB = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_train_RHOB2 = X_train_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_RHOB = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_test_RHOB2 = X_test_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_RHOB = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_hidden_RHOB2 = X_hidden_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  '''
  Predicting RHOB and 
  filling empty spaces by predictions.

  '''

  #Predicting DTS (COMPLETE DATASETS)
  traindata['RHOB_pred'] = model4000.predict(X_train_RHOB2)
  testdata['RHOB_pred'] = model4000.predict(X_test_RHOB2)
  hiddendata['RHOB_pred'] = model4000.predict(X_hidden_RHOB2)

  #Inputing nan values in DTS with DTS_PREDICTED
  traindata['RHOB_COMB'] = traindata['RHOB']
  traindata['RHOB_COMB'].fillna(traindata['RHOB_pred'], inplace=True)

  testdata['RHOB_COMB'] = testdata['RHOB']
  testdata['RHOB_COMB'].fillna(testdata['RHOB_pred'], inplace=True)

  hiddendata['RHOB_COMB'] = hiddendata['RHOB']
  hiddendata['RHOB_COMB'].fillna(hiddendata['RHOB_pred'], inplace=True)




  ##4. PREDICTING DTC
  print('----------------------------------------PREDINCTING DTC-----------------------------------------')
  '''
  Getting data containing DTC readings and 
  spliting into a new training and validation subsets.

  '''
  traindata_dtc = traindata[traindata.DTC.notna()]
  X_dtc = traindata_dtc.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  Y_dtc = traindata_dtc['DTC']

  '''
  Inputing remanent features by their median.

  '''

  X_dtc_inp = X_dtc.apply(lambda x: x.fillna(x.median()), axis=0)                        #Imputation
  X_dtc_train, X_dtc_val, Y_dtc_train, Y_dtc_val = train_test_split(X_dtc_inp, Y_dtc, 
                                                                    test_size=0.3, 
                                                                    random_state=42)     #Spliting train-validation

  print('Training set sahpe {} and validation set shape {}'.format(X_dtc_train.shape, X_dtc_val.shape))
    
  '''
  Runnning XGB Regressor

  '''

  model3000 = XGBRegressor(tree_method='gpu_hist')
  
  model3000.fit(X_dtc_train, Y_dtc_train.values.ravel(), early_stopping_rounds=100, eval_set=[(X_dtc_val, Y_dtc_val)], verbose=100)
  train_pred = model3000.predict(X_dtc_train)
  val_pred = model3000.predict(X_dtc_val)

  print('Train set root mean squared error:', np.sqrt(mean_squared_error(Y_dtc_train, train_pred)))
  print('Validation set root mean squared error:', np.sqrt(mean_squared_error(Y_dtc_val, val_pred)))

  print('--------------------------------Imputing DTC LOG by ML predictions--------------------------------')

  '''
  Inputing median in the whole sets 
  before predinting DTC.

  '''

  # Filling nan values before predicting NPHI
  X_train_DTC = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_train_DTC2 = X_train_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_DTC = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_test_DTC2 = X_test_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_DTC = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_hidden_DTC2 = X_hidden_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  #Predicting DTS (COMPLETE DATASETS)
  traindata['DTC_pred'] = model3000.predict(X_train_DTC2)
  testdata['DTC_pred'] = model3000.predict(X_test_DTC2)
  hiddendata['DTC_pred'] = model3000.predict(X_hidden_DTC2)
  
  '''
  Predicting DTC and 
  filling empty spaces by predictions.

  '''

  #Inputing nan values in DTS with DTS_PREDICTED
  traindata['DTC_COMB'] = traindata['DTC']
  traindata['DTC_COMB'].fillna(traindata['DTC_pred'], inplace=True)

  testdata['DTC_COMB'] = testdata['DTC']
  testdata['DTC_COMB'].fillna(testdata['DTC_pred'], inplace=True)

  hiddendata['DTC_COMB'] = hiddendata['DTC']
  hiddendata['DTC_COMB'].fillna(hiddendata['DTC_pred'], inplace=True)

  #additional features
  print('--------------------------------Creating additional features--------------------------------')


  '''
  Creating additional features:
  '''
  #Train Set
  traindata['S_I'] = traindata.RHOB * (1e6/traindata.DTS_COMB)
  traindata['P_I'] = traindata.RHOB * (1e6/traindata.DTC)
  traindata['DT_R'] = traindata.DTC / traindata.DTS_COMB
  traindata['G'] = ((1e6/traindata.DTS_COMB)**2) * traindata.RHOB
  traindata['K'] = (((1e6/traindata.DTC)**2) * traindata.RHOB) - (4 * traindata.G/3)
  traindata['MD_TVD'] = -(traindata.DEPTH_MD/traindata.Z_LOC)

  #Test Set
  testdata['S_I'] = testdata.RHOB * (1e6/testdata.DTS_COMB)
  testdata['P_I'] = testdata.RHOB * (1e6/testdata.DTC)
  testdata['DT_R'] = testdata.DTC / testdata.DTS_COMB
  testdata['G'] = ((1e6/testdata.DTS_COMB)**2) * testdata.RHOB
  testdata['K'] = (((1e6/testdata.DTC)**2) * testdata.RHOB) - (4 * testdata.G/3)
  testdata['MD_TVD'] = -(testdata.DEPTH_MD/testdata.Z_LOC)

  #Hidden Set
  hiddendata['S_I'] = hiddendata.RHOB * (1e6/hiddendata.DTS_COMB)
  hiddendata['P_I'] = hiddendata.RHOB * (1e6/hiddendata.DTC)
  hiddendata['DT_R'] = hiddendata.DTC / hiddendata.DTS_COMB
  hiddendata['G'] = ((1e6/hiddendata.DTS_COMB)**2) * hiddendata.RHOB
  hiddendata['K'] = (((1e6/hiddendata.DTC)**2) * hiddendata.RHOB) - (4 * hiddendata.G/3)
  hiddendata['MD_TVD'] = -(hiddendata.DEPTH_MD/hiddendata.Z_LOC)

  #dropping unnecessary data
  #cols_drop = ['DTS_pred', 'NPHI_pred', 'RHOB_pred', 'DTC_pred']
  #traindata = traindata.drop(cols_drop, axis=1)
  #testdata = testdata.drop(cols_drop, axis=1)
  #hiddendata = hiddendata.drop(cols_drop, axis=1)

  print('Features included in the datasets: {}'.format(traindata.columns))

  return traindata, testdata, hiddendata