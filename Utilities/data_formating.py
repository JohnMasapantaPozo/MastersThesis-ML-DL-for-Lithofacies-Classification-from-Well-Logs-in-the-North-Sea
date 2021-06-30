
def formating(training_raw, test_raw, hidden_raw):
  '''

  Data columns are renamed, lithofacies classes are mapped to values
  from 0 to 11, and intepretation confidence is dropped.

  '''

  lithology_numbers = {30000: 0, 
                       65030: 1, 
                       65000: 2, 
                       80000: 3, 
                       74000: 4, 
                       70000: 5, 
                       70032: 6, 
                       88000: 7, 
                       86000: 8, 
                       99000: 9, 
                       90000: 10, 
                       93000: 11
                       }

  #formating raw training set
  training_formated = training_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  training_formated = training_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  training_formated['LITHO'] = training_formated["LITHO"].map(lithology_numbers)

  #formating raw test set
  test_formated = test_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  test_formated['LITHO'] = test_formated["LITHO"].map(lithology_numbers)

  #formating raw hidden set
  hidden_formated = hidden_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  hidden_formated = hidden_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  hidden_formated['LITHO'] = hidden_formated['LITHO'].map(lithology_numbers)

  return(training_formated, test_formated, hidden_formated)