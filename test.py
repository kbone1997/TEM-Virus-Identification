"""
Copyright (c) 2020 Damian Matuszewski
Centre for Image Analysis
Uppsala University
"""

from __future__ import print_function

from shutil import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import csv
from data import load_test_data, CLASSES_NO, CLASSES_NAMES
from train import get_models

MODEL_NAME = 'CUSTOM'
WEIGHTS_VERSION = '_ACCURACY_BEST' # '_LOSS_BEST' '_ACCURACY_BEST'
TRANSFER_LEARNING = False


#------------------------------------------------------------------------------
def confusion_matrix_to_CSV(cm, target_names, fileName):
    with open(fileName, 'w', newline='') as csvCM:
        writer = csv.writer(csvCM)
        writer.writerow(['']+target_names)
        for i in range(CLASSES_NO):
            writer.writerow([target_names[i]]+list(cm[i]))
            

#------------------------------------------------------------------------------
if __name__ == '__main__':
    X,Y = load_test_data()
    
    if TRANSFER_LEARNING:
        classifier, OOD = transfer_model() 
        
    else:
        classifier, OOD = get_models()
    
    modelDir = 'models/' + MODEL_NAME + '/'
    copy('test.py', modelDir+'test.py')
    copy('data.py', modelDir+'data_test.py')
    
    classifier.load_weights(modelDir+MODEL_NAME+'_weights'+WEIGHTS_VERSION+'.h5')
    OOD.load_weights(modelDir+MODEL_NAME+'_weights'+WEIGHTS_VERSION+'.h5')
    
    
    if TRANSFER_LEARNING:
        X = np.concatenate((X,X,X),-1)
    
    classifierPrediction = classifier.predict(X)
    OODPrediction = OOD.predict(X)
    
    for i in [10,200,300]: #,50,70,90,110,130,150]:
        print('-'*30)    
        for j in range(CLASSES_NO):
            print('{:.0f}\t{:.3f}\t{:.3f}\n'.format(Y[i][j], classifierPrediction[i][j], OODPrediction[i][j]))
    
    #Confution Matrix and Classification Report
    y_pred = np.argmax(classifierPrediction, axis=1)
    y_true = np.argmax(Y, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fn = 'models/'+MODEL_NAME+'/'+MODEL_NAME+'_confusion_matrix_TEST'+WEIGHTS_VERSION+'.csv'
    confusion_matrix_to_CSV(cm, CLASSES_NAMES, fn)
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=CLASSES_NAMES))
    cr_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASSES_NAMES, digits = 3, output_dict=True)).transpose()
    cr_df.to_csv('models/'+MODEL_NAME+'/'+MODEL_NAME+'_classification_report_TEST'+WEIGHTS_VERSION+'.csv', index= True)