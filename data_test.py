"""
Copyright (c) 2020 Damian Matuszewski
Centre for Image Analysis
Uppsala University
"""

from __future__ import print_function
import glob
import os
import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from sklearn.utils import shuffle

# if TENSORFLOW -> use channels_last
# if THEANO -> use channels_first

train_data_path = 'context virus/augmented_train/'
test_data_path =  'context virus/test/'
val_data_path = 'context virus/validation/'
# train_data_path = 'context virus/WORST CASE/augmented_train_1nm/'    
# test_data_path =  'context virus/WORST CASE/test_1nm/' 
# val_data_path = 'context virus/WORST CASE/validation_1nm/' 

CLASSES_NO = 14 
TRAIN_SAMPLES = CLASSES_NO*736

IMAGE_ROWS = 256 
IMAGE_COLS = 256 

SHUFFLE_BEFORE_SAVING = False

CLASSES_NAMES = ['Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza', 'Lassa', 'Marburg', 'Nipah', 'Norovirus', 'Orf', 'Papilloma', 'Rift Valley', 'Rotavirus']

# =============================================================================
def normalizeImage(img):
    img = np.array(img)
    img = img_as_float(img)
    img = img.astype('float32')
    mean = np.mean(img)  # mean for data centering
    std = np.std(img)  # std for data normalization
    img -= mean
    img /= std
    return img


# -----------------------------------------------------------------------------
def create_train_data():
    print('-'*30)
    print('Loading and processing training images...')
    
    X = []
    Y = []
    FileNames = [] 
    
    classPaths = glob.glob(train_data_path + '*/')
    if len(classPaths) != CLASSES_NO:
        print('Wrong CLASSES_NO!')
    
    c_idx = 0
    for cp in classPaths:
        oneHotClass = np.eye(CLASSES_NO)[c_idx]
        oneHotClass = oneHotClass.astype('float32')
        c_idx = c_idx+1  
        images = glob.glob(cp + '*.tif')
        if len(images) != TRAIN_SAMPLES/CLASSES_NO:
            print('Wrong TRAIN_SAMPLES!' + cp)
        
        for image_name in images:
            img = imread(image_name)
            img = normalizeImage(img)
            X.append(img)
            Y.append(oneHotClass)
            FileNames.append(image_name)            

        print('Class - ' + cp.split('\\')[-2] + ' - DONE')


    # shuffle the data in a consistent way
    if SHUFFLE_BEFORE_SAVING:
        X, Y = shuffle(X,Y)

    X = np.array(X)
    Y = np.array(Y)
    X = X[..., np.newaxis] # add explicit channel dimension

    file_name = train_data_path + '/X_train.npy' 
    np.save(file_name, X)
    file_name = train_data_path + '/Y_train.npy' 
    np.save(file_name, Y)
    with open(train_data_path + '/FileNames_train.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % fn for fn in FileNames)
  
    print('Loading done.')
    print('-'*30)

    
# -----------------------------------------------------------------------------
def load_train_data():
    if os.path.isfile(train_data_path + '/X_train.npy'):
        X = np.load(train_data_path + '/X_train.npy')
    if os.path.isfile(train_data_path + '/Y_train.npy'):
        Y = np.load(train_data_path + '/Y_train.npy')
    return X, Y

def load_train_data_fileNames():
    with open(train_data_path + '/FileNames_train.txt', 'r') as filehandle:
        FileNames = [fn.rstrip() for fn in filehandle.readlines()]
    return FileNames

# -----------------------------------------------------------------------------
def create_validation_data():
    print('-'*30)
    print('Loading and processing validation images...')
    
    X = []
    Y = []
    FileNames = [] 
     
     
    classPaths = glob.glob(val_data_path + '*/')[0:-1]
    if len(classPaths) != CLASSES_NO:
        print('Wrong CLASSES_NO!')
    
    c_idx = 0
    for cp in classPaths:
        oneHotClass = np.eye(CLASSES_NO)[c_idx]
        oneHotClass = oneHotClass.astype('float32')
        c_idx = c_idx+1  
        images = glob.glob(cp + '*.tif')
        
        for image_name in images:
            img = imread(image_name)
            img = normalizeImage(img)
            X.append(img)
            Y.append(oneHotClass)
            FileNames.append(image_name)  
            
        print('Class - ' + cp.split('\\')[-2] + ' - DONE')

    X = np.array(X)
    Y = np.array(Y)
    X = X[..., np.newaxis] # add explicit channel dimension

    file_name = val_data_path + '/X_validation.npy' 
    np.save(file_name, X)
    file_name = val_data_path + '/Y_validation.npy' 
    np.save(file_name, Y)
    with open(val_data_path + '/FileNames_validation.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % fn for fn in FileNames)
  
    print('Loading done.')
    print('-'*30)

    
# -----------------------------------------------------------------------------
def load_validation_data():
    if os.path.isfile(val_data_path + '/X_validation.npy'):
        X = np.load(val_data_path + '/X_validation.npy')
    if os.path.isfile(val_data_path + '/Y_validation.npy'):
        Y = np.load(val_data_path + '/Y_validation.npy')
    return X, Y

def load_validation_data_fileNames():
    with open(val_data_path + '/FileNames_validation.txt', 'r') as filehandle:
        FileNames = [fn.rstrip() for fn in filehandle.readlines()]
    return FileNames
        
# -----------------------------------------------------------------------------
def create_test_data():
    print('-'*30)
    print('Loading and processing test images...')
    
    X = []
    Y = []
    FileNames = [] 
        
    classPaths = glob.glob(test_data_path + '*/')[0:-1]
    if len(classPaths) != CLASSES_NO:
        print('Wrong CLASSES_NO!')
    
    c_idx = 0
    for cp in classPaths:
        oneHotClass = np.eye(CLASSES_NO)[c_idx]
        oneHotClass = oneHotClass.astype('float32')
        c_idx = c_idx+1  
        images = glob.glob(cp + '*.tif')
        
        for image_name in images:
            img = imread(image_name)
            img = normalizeImage(img)
            X.append(img)
            Y.append(oneHotClass)
            FileNames.append(image_name)  

        print('Class - ' + cp.split('\\')[-2] + ' - DONE')

    X = np.array(X)
    Y = np.array(Y)
    X = X[..., np.newaxis] # add explicit channel dimension

    file_name = test_data_path + '/X_test.npy' 
    np.save(file_name, X)
    file_name = test_data_path + '/Y_test.npy' 
    np.save(file_name, Y)
    with open(test_data_path + 'FileNames_test.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % fn for fn in FileNames)
        
    print('Loading done.')
    print('-'*30)

    
# -----------------------------------------------------------------------------
def load_test_data():
    if os.path.isfile(test_data_path + '/X_test.npy'):
        X = np.load(test_data_path + '/X_test.npy')
    if os.path.isfile(test_data_path + '/Y_test.npy'):
        Y = np.load(test_data_path + '/Y_test.npy')
    return X, Y

def load_test_data_fileNames():
    with open(test_data_path + '/FileNames_test.txt', 'r') as filehandle:
        FileNames = [fn.rstrip() for fn in filehandle.readlines()]
    return FileNames

# =============================================================================
if __name__ == '__main__':
    create_train_data()
    X,Y = load_train_data()
    FN = load_train_data_fileNames()
    
    create_validation_data()
    X_val, Y_val = load_validation_data()
    FN_val = load_validation_data_fileNames()
    
    create_test_data()
    X_test, Y_test = load_test_data()
    FN_test = load_test_data_fileNames()    
    
