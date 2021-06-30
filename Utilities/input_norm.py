
def normalization(traindata, testdata, hiddendata):
  import pandas as pd
  from sklearn.preprocessing import StandardScaler

  '''

  The features that were not augmented
  by ML are inputed by the median and then standardized.

  '''

  train_features = traindata.drop(['LITHO'], axis=1);   train_labels = traindata['LITHO']
  test_features = testdata.drop(['LITHO'], axis=1);     test_labels = testdata['LITHO']
  hidden_features = hiddendata.drop(['LITHO'], axis=1); hidden_labels = hiddendata['LITHO']

  #Imputng features by median
  train_features_inp = train_features.apply(lambda x: x.fillna(x.median()), axis=0)
  test_features_inp = test_features.apply(lambda x: x.fillna(x.median()), axis=0)
  hidden_features_inp = hidden_features.apply(lambda x: x.fillna(x.median()), axis=0)

  n = train_features_inp.shape[1]
  std = StandardScaler()
  x_train_std = train_features_inp.copy()
  x_test_std = test_features_inp.copy()
  x_hidden_std = hidden_features_inp.copy()

  x_train_std.iloc[:,:n] = std.fit_transform(x_train_std.iloc[:,:n])
  x_test_std.iloc[:,:n] = std.transform(x_test_std.iloc[:,:n])
  x_hidden_std.iloc[:,:n] = std.transform(x_hidden_std.iloc[:,:n])

  #Concatenating normalized features and labels
  cleaned_traindata = pd.concat([x_train_std, train_labels], axis=1)
  cleaned_testdata = pd.concat([x_test_std, test_labels], axis=1)
  cleaned_hiddendata = pd.concat([x_hidden_std, hidden_labels], axis=1)

  return cleaned_traindata, cleaned_testdata, cleaned_hiddendata