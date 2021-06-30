
def run_SVM(train_norm, test_norm, hidden_norm):
  from sklearn.model_selection import train_test_split
  from sklearn.svm import SVC

  x_train = train_norm.drop(['LITHO'], axis=1)
  y_train = train_norm['LITHO']

  x_test = test_norm.drop(['LITHO'], axis=1)
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm.drop(['LITHO'], axis=1)
  y_hidden = hidden_norm['LITHO']

  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train, y_train, train_size=0.1, shuffle=True, stratify=y_train, random_state=0)

  #Base Model
  model_svm = SVC(kernel='rbf',
                  C=0.5,
                  #cache_size=5000,
                  #decision_function_shape='ovr'
                  )
  
  model_svm.fit(x_train_strat, y_train_strat)

  train_pred_svm = model_svm.predict(x_train)
  test_pred_svm = model_svm.predict(x_test)
  hidden_pred_svm = model_svm.predict(x_hidden)

  return train_pred_svm, test_pred_svm, hidden_pred_svm