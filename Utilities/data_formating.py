
"""Data Formating

This script simply rename the column names in a consitent manner for the training, 
open test, and hidden test sets. It also maps the lithofacies labesl with numbers
from 0 to 11 and drops the intepretation confidence column.
"""

def formating(training_raw, test_raw, hidden_raw):

  """Returns the training, open test, and hidden test dataframes with consistent formats.

  Parameters
  ----------
  training_raw: Dataframe
    Raw training dataframe.
  test_raw: Dataframe
    Raw open test dataframe.
  hidden_raw: Dataframe
    Raw hidden test dataframe.

  Returns
  ----------
  training_formated: Dataframe
    Formated training dataframe.
  test_formated: Dataframe
    Formated open test dataframe.
  hidden_formated: Dataframe
    Formated hidden test dataframe.
  """

  lithology_numbers = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 
                       70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 
                       93000: 11
                       }

  # formating raw training set
  training_formated = training_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  training_formated = training_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  training_formated['LITHO'] = training_formated["LITHO"].map(lithology_numbers)

  #formating raw test set
  test_formated = test_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  test_formated['LITHO'] = test_formated["LITHO"].map(lithology_numbers)

  # formating raw hidden set
  hidden_formated = hidden_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  hidden_formated = hidden_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  hidden_formated['LITHO'] = hidden_formated['LITHO'].map(lithology_numbers)

  return(training_formated, test_formated, hidden_formated)