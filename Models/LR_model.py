
def run_LR(train_norm, test_norm, hidden_norm):
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression

  #Stratified Sample
  features_selected_lr = ['DTS_COMB', 'G', 'P_I', 'GR', 
                          'NPHI_COMB', 'DTC', 'RHOB', 'DT_R', 
                          'Z_LOC', 'S_I','K'
                          ]

  x_train = train_norm[features_selected_lr]
  y_train = train_norm['LITHO']

  x_test = test_norm[features_selected_lr]
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm[features_selected_lr]
  y_hidden = hidden_norm['LITHO']
  
  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train, y_train, train_size=0.1, shuffle=True, stratify=y_train, random_state=0)

  #Base Model
  model_lr = LogisticRegression(C=0.1, solver='saga', max_iter=4000, verbose=1)
  model_lr.fit(x_train_strat[features_selected_lr], y_train_strat)

  train_pred_lr = model_lr.predict(x_train[features_selected_lr])
  test_pred_lr = model_lr.predict(x_test[features_selected_lr])
  hidden_pred_lr = model_lr.predict(x_hidden[features_selected_lr])

  return train_pred_lr, test_pred_lr, hidden_pred_lr