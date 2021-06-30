
def run_DT(train_norm, test_norm, hidden_norm):
  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeClassifier

  x_train = train_norm.drop(['LITHO'], axis=1)
  y_train = train_norm['LITHO']

  x_test = test_norm.drop(['LITHO'], axis=1)
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm.drop(['LITHO'], axis=1)
  y_hidden = hidden_norm['LITHO']
  
  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train, y_train, train_size=0.1, shuffle=True, stratify=y_train, random_state=0)

  tunned_dt = DecisionTreeClassifier(max_depth=15,
                                     ccp_alpha=0.002
                                     )

  # Fit the regressor to the training data
  tunned_dt.fit(x_train_strat, y_train_strat)

  train_pred_dtp = tunned_dt.predict(x_train)
  open_pred_dtp = tunned_dt.predict(x_test)
  hidden_pred_dtp = tunned_dt.predict(x_hidden)

  return train_pred_dtp, open_pred_dtp, hidden_pred_dtp