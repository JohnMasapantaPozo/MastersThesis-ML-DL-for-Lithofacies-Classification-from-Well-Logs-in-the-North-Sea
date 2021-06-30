
def run_KNN(train_norm, test_norm, hidden_norm):
  from sklearn.model_selection import train_test_split
  from sklearn import neighbors

  #Stratified Sample
  selectedfeatures_knn = ['GR', 'FORMATION_encoded', 'GROUP_encoded', 'NPHI_COMB', 'RHOB', 
                            'X_LOC', 'BS', 'CALI', 'SP', 'WELL_encoded', 'Z_LOC', 'DT_R', 'DEPTH_MD',
                            'DTC', 'Cluster']

  x_train = train_norm[selectedfeatures_knn]
  y_train = train_norm['LITHO']

  x_test = test_norm[selectedfeatures_knn]
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm[selectedfeatures_knn]
  y_hidden = hidden_norm['LITHO']

  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train, y_train, train_size=0.1, shuffle=True, stratify=y_train, random_state=0)

  #Base Model
  model_knn = neighbors.KNeighborsClassifier(n_neighbors=80, 
                                              weights='distance', 
                                              metric='manhattan'
                                              )
  model_knn.fit(x_train_strat[selectedfeatures_knn], y_train_strat)

  train_pred_knn = model_knn.predict(x_train[selectedfeatures_knn])
  test_pred_knn = model_knn.predict(x_test[selectedfeatures_knn])
  hidden_pred_knn = model_knn.predict(x_hidden[selectedfeatures_knn])

  return train_pred_knn, test_pred_knn, hidden_pred_knn