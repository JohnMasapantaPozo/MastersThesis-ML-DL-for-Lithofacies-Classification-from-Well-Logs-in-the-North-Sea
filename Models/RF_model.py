
def run_RF(train_norm, test_norm, hidden_norm):
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  features_selected_rf = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'P_I', 'S_I', 'DTC', 'DTS_COMB',
                        'RSHA', 'DT_R', 'RHOB', 'K', 'DCAL', 'Y_LOC', 'GROUP_encoded',
                        'WELL_encoded', 'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI',
                        'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD', 'ROP', 'DRHO']

  x_train = train_norm[features_selected_rf]
  y_train = train_norm['LITHO']

  x_test = test_norm[features_selected_rf]
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm[features_selected_rf]
  y_hidden = hidden_norm['LITHO']

  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train, y_train, train_size=0.5, shuffle=True, stratify=y_train, random_state=0)

  # PREDICTION ON THE VALIDATION SET
  model_rf = RandomForestClassifier(n_estimators=350,
                                    bootstrap=False,
                                    max_depth=45,
                                    max_features='sqrt'
                                    )

  # Fit the regressor to the training data
  model_rf.fit(x_train_strat[features_selected_rf], y_train_strat.values.ravel())

  # Prediction
  train_pred_rf = model_rf.predict(x_train[features_selected_rf])
  open_pred_rf = model_rf.predict(x_test[features_selected_rf])
  hidden_pred_rf = model_rf.predict(x_hidden[features_selected_rf])

  return train_pred_rf, open_pred_rf, hidden_pred_rf