
# Evaluate prediction


def matrix_score(y_true, y_pred):
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
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

    from itertools import product

    #facies_dict = {0:'Sandstone', 1:'Sandstone/Shale', 2:'Shale', 3:'Marl',
                   #4:'Dolomite', 5:'Limestone', 6:'Chalk', 7:'Halite', 
                   #8:'Anhydrite', 9:'Tuff', 10:'Coal', 11:'Basement'}
    #labels = np.array([facies_dict[k] for k in np.sort(y_pred.unique())])
    labels = ['Sandstone', 'Sandstone/Shale', 'Shale', 'Marl',
              'Dolomite','Limestone','Chalk','Halite', 
              'Anhydrite', 'Tuff','Coal','Basement']

    fig, ax = plt.subplots(figsize=(10,8))
    #Normalizing confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    summ = cm.sum(axis=0)
    cm_norm = pd.DataFrame(np.zeros(cm.shape))
    for i in range(cm.shape[1]):
      for j in range(cm.shape[0]):
        cm_norm[i][j] = cm[i][j]*100/summ[i]

    cm_final = cm_norm.to_numpy()

    #disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=labels)
    plt.imshow(cm_final, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('NORMALIZED CONFUSION MATRIX', size=12)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.colorbar()
    
    fmt = '.2f'
    thresh = cm_final.max() / 2.
    for i, j in product(range(cm_final.shape[0]),   range(cm_final.shape[1])):
        plt.text(j, i, format(cm_final[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_final[i, j] > thresh else "black")
        
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)