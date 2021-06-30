
def run_LightBoost(train_norm, test_norm, hidden_norm):

  from lightgbm import LGBMClassifier

  selected_features_lightboost = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'DTC', 'DTS_COMB', 'RSHA', 'DT_R',
                                'RHOB', 'K', 'DCAL', 'Y_LOC', 'GROUP_encoded', 'WELL_encoded',
                                'DEPTH_MD', 'Z_LOC', 'CALI', 'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD',
                                'ROP', 'DRHO']
                 
  x_train = train_norm[selected_features_lightboost]
  y_train = train_norm['LITHO']

  x_test = test_norm[selected_features_lightboost]
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm[selected_features_lightboost]
  y_hidden = hidden_norm['LITHO']

  '''

  The model is trained on 10 stratified k-folds, also uses 
  the open set as validation set to avoid overfitting
  and a 100-round early stopping callback.

  The model uses a multi-soft_probability objective function
  which returns the probabilities predicted for each class.
  This probabilities are computed and stacked by using each k-fold
  to give the final prediction.

  '''

  lightboost_model1 = LGBMClassifier(n_estimators=1000, learning_rate=0.015, 
                                  random_state=42, max_depth=8, 
                                  reg_lambda=250, verbose=-1,
                                  objective='multi:softprob',
                                  device='gpu', gpu_platform_id=1,
                                  gpu_device_id=0, silent=True
                                  )

  lightboost_model1.fit(x_train, y_train.values.ravel(), early_stopping_rounds=100, eval_set=[(x_test, y_test)], verbose=-100, )
  train_pred_light = lightboost_model1.predict(x_train)
  open_pred_light = lightboost_model1.predict(x_test)
  hidden_pred_light = lightboost_model1.predict(x_hidden)

  return train_pred_light, open_pred_light, hidden_pred_light
  