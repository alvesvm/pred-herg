import pandas as pd
from sklearn import metrics

def stats(y_train, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_train, y_pred)
    accuracy = metrics.accuracy_score(y_train, y_pred)
#    roc_auc_score = metrics.roc_auc_score(y_train, y_pred)
    Kappa = metrics.cohen_kappa_score(y_train, y_pred, weights='linear')
    # True and false values
    TN, FP, FN, TP = confusion_matrix.ravel()  
    # Sensitivity, hit rate, recall, or true positive rate
    SE = TP/(TP+FN)
    # Specificity or true negative rate
    SP = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Correct classification rate
    CCR = (SE + SP)/2
    d = dict({'Accuracy': accuracy,
#         'AUC': roc_auc_score,
         'Kappa': Kappa,
         'CCR': CCR,
         'Sensitivity': SE,
         'PPV': PPV,
         'Specificity': SP,
         'NPV': NPV})
    return pd.DataFrame(d, columns=d.keys(), index=[0]).round(2)
