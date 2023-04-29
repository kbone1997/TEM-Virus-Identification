"""
Copyright (c) 2020 Damian Matuszewski
Centre for Image Analysis
Uppsala University
"""

from __future__ import print_function

import os
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras import initializers
from keras import metrics
from keras.regularizers import l1, l2, l1_l2
from keras.applications import VGG16, VGG19, Xception, ResNet50V2, InceptionV3, MobileNetV2, DenseNet201,InceptionResNetV2
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import csv
from data import load_train_data, load_validation_data, IMAGE_ROWS, IMAGE_COLS, CLASSES_NO, CLASSES_NAMES

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
AXIS = -1 # -1 for 'channels_last' and 0 for 'channels_first'

MODEL_NAME = 'DenseNet201_1024_50ep'
TRANSFER_LEARNING = True
EPOCHS_NO = 50
BATCH_SIZE = 12
W_SEED = list(range(40)) # None

reg_kernel = l2(l=0.1)

#------------------------------------------------------------------------------
def get_models():
    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[1]), kernel_regularizer=reg_kernel, name='conv1_1')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(63, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[2]), kernel_regularizer=reg_kernel, name='conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = concatenate([conv1, inputs], axis=AXIS)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(rate=0.2)(pool1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[3]), kernel_regularizer=reg_kernel, name='conv2_1')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[4]), kernel_regularizer=reg_kernel, name='conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = concatenate([conv2, pool1], axis=AXIS)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(rate=0.25)(pool2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[5]), kernel_regularizer=reg_kernel, name='conv3_1')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[6]), kernel_regularizer=reg_kernel, name='conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = concatenate([conv3, pool2], axis=AXIS)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(rate=0.3)(pool3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[7]), kernel_regularizer=reg_kernel, name='conv4_1')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[8]), kernel_regularizer=reg_kernel, name='conv4_2')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = concatenate([conv4, pool3], axis=AXIS)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(rate=0.35)(pool4)

    # conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[9]), name='conv5_1')(pool4)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[10]), name='conv5_2')(conv5)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = concatenate([conv5, pool4], axis=AXIS)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # pool5 = Dropout(rate=0.35)(pool5)

    # conv6 = Conv2D(256, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[11]), name='conv6_1')(pool5)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Conv2D(256, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.glorot_normal(W_SEED[12]), name='conv6_2')(conv6)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = concatenate([conv6, pool5], axis=AXIS)
    # pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    # pool6 = Dropout(rate=0.25)(pool6)

    pool6 = Conv2D(256, (3, 3), activation='relu', padding='valid', kernel_initializer=initializers.glorot_normal(W_SEED[21]), kernel_regularizer=reg_kernel, name='conv_LAST')(pool4)
    flat = Flatten()(pool6)
    
    dense = Dense(256, activation='relu', kernel_initializer=initializers.glorot_normal(W_SEED[31]), kernel_regularizer=reg_kernel)(flat)
    dense = Dropout(rate=0.35)(dense)  
    dense = Dense(256, activation='relu', kernel_initializer=initializers.glorot_normal(W_SEED[32]), kernel_regularizer=reg_kernel)(dense)   
    output = Dense(CLASSES_NO, activation=None, kernel_initializer=initializers.glorot_normal(W_SEED[33]))(dense)
    
    result = Activation(activation='softmax')(output)
    
    OOD = Model(inputs=[inputs], outputs=[output])
    classifier = Model(inputs=[inputs], outputs=[result])
    classifier.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy", metrics=[metrics.categorical_accuracy])
    return classifier, OOD

#------------------------------------------------------------------------------
def transfer_model():
    model = DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_ROWS, IMAGE_COLS, 3),
        classes=14
    )
     
    # add new classifier layers
    flat = Flatten()(model.layers[-1].output)
    feat1 = Dense(1024, activation='relu')(flat)
    feat1 = Dropout(rate=0.3)(feat1) 
    feat2 = Dense(1024, activation='relu')(feat1)
    feat2 = Dropout(rate=0.3)(feat2)   
    output = Dense(CLASSES_NO, activation=None,)(feat2)
    result = Activation(activation='softmax')(output)

    OOD = Model(inputs=model.inputs, outputs=output)
    classifier = Model(inputs=model.inputs, outputs=result)
    classifier.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy", metrics=[metrics.categorical_accuracy])
    return classifier, OOD
    

#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------
def train_CNN(x, y, x_v, y_v, modelName):
    print('\nTraining model ' + modelName +'\n')
    
    print('-'*30)
    print('Creating and compiling model ' + modelName + '...')
    print('-'*30)
    
    if TRANSFER_LEARNING:
        classifier, OOD = transfer_model()
    else:
        classifier, OOD = get_models()
    classifier.summary()
    
    modelDir = 'models/' + modelName
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    copy('data.py', modelDir+'/data.py')
    copy('train.py', modelDir+'/train.py')
    
    # callbacks
    modelCheckpoint = ModelCheckpoint(modelDir+'/'+modelName+'_weights_ACCURACY_BEST.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max')
    modelCheckpointLoss = ModelCheckpoint(modelDir+'/'+modelName+'_weights_LOSS_BEST.h5', monitor='val_loss', save_best_only=True, mode='min')    
    csvLogger = CSVLogger(modelDir+'/'+modelName+'_log.csv', append=True)
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, epsilon=0.001, min_lr=1e-7)
 
    print('-'*30)
    print('Fitting model ' + modelName + '...')
    print('-'*30)
    
    history = classifier.fit(x, y, validation_data=(x_v,y_v), epochs=EPOCHS_NO, 
                             batch_size=BATCH_SIZE, shuffle = True, 
                             callbacks=[modelCheckpoint, modelCheckpointLoss, csvLogger]) 
    
    
    print('-'*30)
    print('Plotting and saving model ' + modelName + ' history...')
    print('-'*30)
    
    # list all data in history
    print(history.history.keys())
    # plot history for accuracy
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    maxAcc = max(history.history['val_categorical_accuracy'])
    maxAccEp = np.argmax(history.history['val_categorical_accuracy'])
    plt.plot(maxAccEp, maxAcc, 'k.')
    plt.title('Model \"'+modelName+'\" - accuracy - best = '+str(round(maxAcc,3))+' (val)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    figFileName = modelDir+'/'+modelName
    fig.savefig(figFileName+'_accuracy.pdf', dpi=720, bbox_inches='tight')
    fig.savefig(figFileName+'_accuracy.png', dpi=720, bbox_inches='tight')
    
    # plot history for loss
    fig = plt.figure() 
    fig.set_facecolor('white')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    minLoss = min(history.history['val_loss'])
    minLossEp = np.argmin(history.history['val_loss'])
    plt.plot(minLossEp, minLoss, 'k.')
    plt.title('Model \"'+modelName+'\" - loss - best = '+str(round(minLoss,3))+' (val)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    fig.savefig(figFileName+'_loss.pdf', dpi=720, bbox_inches='tight')
    fig.savefig(figFileName+'_loss.png', dpi=720, bbox_inches='tight')
    
    classifier.load_weights(modelDir+'/'+modelName+'_weights_ACCURACY_BEST.h5')
    OOD.load_weights(modelDir+'/'+modelName+'_weights_ACCURACY_BEST.h5')
    
    return classifier, OOD

#------------------------------------------------------------------------------
def confusion_matrix_to_CSV(cm, target_names, fileName):
    with open(fileName, 'wt') as csvCM:
        writer = csv.writer(csvCM)
        writer.writerow(['']+target_names)
        for i in range(CLASSES_NO):
            writer.writerow([target_names[i]]+list(cm[i]))
            

#------------------------------------------------------------------------------
if __name__ == '__main__':
    #print("Num GPUs Available: ", len(list_physical_devices('GPU')))
    
    X,Y = load_train_data()
    X_val, Y_val = load_validation_data()
    
    if TRANSFER_LEARNING:
        X = np.concatenate((X,X,X),-1)
        X_val = np.concatenate((X_val,X_val,X_val),-1)
        
    classifier, OOD = train_CNN(X, Y, X_val, Y_val, MODEL_NAME)
    
    classifierPrediction = classifier.predict(X_val)
    OODPrediction = OOD.predict(X_val)
    
    for i in [10,200,300]: #,50,70,90,110,130,150]:
        print('-'*30)    
        for j in range(CLASSES_NO):
            print('{:.0f}\t{:.3f}\t{:.3f}\n'.format(Y_val[i][j], classifierPrediction[i][j], OODPrediction[i][j]))
    
    #Confution Matrix and Classification Report
    y_pred = np.argmax(classifierPrediction, axis=1)
    y_true = np.argmax(Y_val, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fn = 'models/'+MODEL_NAME+'/'+MODEL_NAME+'_confusion_matrix.csv'
    confusion_matrix_to_CSV(cm, CLASSES_NAMES, fn)
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=CLASSES_NAMES))
    cr_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASSES_NAMES, digits = 3, output_dict=True)).transpose()
    cr_df.to_csv('models/'+MODEL_NAME+'/'+MODEL_NAME+'_classification_report.csv', index= True)