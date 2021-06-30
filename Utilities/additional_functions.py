
"""Additional functions

This script holds the matrix_score and confusion_matrix functions, which serves as 
evaluation for the classification performance each machine-learning has.

They require some functionalities from libraries such as  pandas, numpy, matplotlib,
and scikit-lean. 
"""

def matrix_score(y_true, y_pred):
    
    """Returns the penalty matrix score obined by the predicted lithofacies a
    particular machine-learning model is able to provide. The matrix score was a metric 
    measure proposed by the FORCE commitee in order to provide the prediction performance
    measure from a petrophyicist perpective.

    Parameters
    ----------
    y_true: list
      The actual lithologies given by the datasets provider.
    y_pred: list
      The predicted lithofacies obtained by a particular machine learning model.

    Returns
    ----------
    matrix penaty score:
      Penalty matrix score obined by a particular machine-learning model.
    """
    
    import numpy as np
    matrix_path = '/content/drive/MyDrive/Thesis_data/penalty_matrix.npy'
    A = np.load(matrix_path)
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]
    
# Confusion Matrix Function

def confusion_matrix(y_true, y_pred):
      
    """Plots a confusion matrix normalized by the number of predictions a particular
    machine learning algorithm has. By ormalize we look at the number of predictions
    the model gets right.

    Parameters
    ----------
    y_true: list
      The actual lithologies given by the datasets provider.
    y_pred: list
      The predicted lithofacies obtained by a particular machine learning model.

    Returns
    ----------
    confusion matrix:
      A normalized confusion matrix by the number of predictions. 
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
    from itertools import product

    def litho_confusion_matrix(y_true, y_pred):
      facies_dict = {0:'Sandstone', 1:'Sandstone/Shale', 2:'Shale', 3:'Marl',
                    4:'Dolomite', 5:'Limestone', 6:'Chalk', 7:'Halite', 
                    8:'Anhydrite', 9:'Tuff', 10:'Coal', 11:'Basement'}

      # creating a lithofacies names
      labels = list(set(list(y_pred.unique()) + list(y_true.unique())))
      label_names = [facies_dict[k] for k in labels]

      # normalizing confusion matrix by the number of predictions
      cm = pd.DataFrame(confusion_matrix(y_true.values, y_pred.values))
      summ = cm.sum(axis=0)
      cm_norm = pd.DataFrame(np.zeros(cm.shape))
      for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
          cm_norm[i][j] = cm[i][j]*100/summ[i]
      cm_final = cm_norm.fillna(0).to_numpy()

      fig, ax = plt.subplots(figsize=(12,8))
      plt.imshow(cm_final, interpolation='nearest', cmap=plt.cm.Blues)
      plt.title('NORMALIZED CONFUSION MATRIX', size=15)
      tick_marks = np.arange(len(label_names))
      plt.xticks(tick_marks, label_names, rotation=90)
      plt.yticks(tick_marks, label_names)
      plt.colorbar()
      
      # creating a scores format (black and white)
      fmt = '.2f'
      thresh = cm_final.max() / 2.
      for i, j in product(range(cm_final.shape[0]),   range(cm_final.shape[1])):
        plt.text(j, i, format(cm_final[i, j], fmt),
                      horizontalalignment="center",
                      color="white" if cm_final[i, j] > thresh else "black")
              
      plt.ylabel('True label', size=14)
      plt.xlabel('Predicted label', size=14)