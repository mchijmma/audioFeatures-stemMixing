
# coding: utf-8

from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np 
import pandas as pd



import sys

import os

import pickle

from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, scale
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




# In[2]:

#Constant Variables python
kGroup = sys.argv[1]
kInstrument = sys.argv[1]
kSampleRate = 44100
kN = 100
kType = 'mono'
kFeatures = 1812 #Find a way to not hard-code this.     
path = './Music/Data/MedleyDB/Features/%s/%s/' % (sys.argv[2], kType)

kInstruments = ['bass', 'guitar', 'vocal', 'keys-all']
kLabels = ['b','g','v','k']


def dumpPickle(d, name, _path = path1):

  with open(_path + name, 'wb') as output:
    # Pickle dictionary using protocol 0.
    pickle.dump(d, output)
  print('%s Saved' % (name))
  
def loadPickle(name,_path = path1):  
  # load data from pkl file
  with open(_path + name, "rb") as fp:
      loaded_data1 = pickle.load(fp)
  
  print('%s loaded, %s ' % (name, type(loaded_data1)))
  
  return loaded_data1



p = './Music/Data/MedleyDB/Features/Regression/' 

df_Raw = loadPickle('df_Raw.pkl', _path = p)
X_train = loadPickle('X_train.pkl', _path = p)
X_test = loadPickle('X_test.pkl', _path = p)
Xs_train = loadPickle('Xs_train.pkl', _path = p)
Xs_test = loadPickle('Xs_test.pkl', _path = p)
Xm_train = loadPickle('Xm_train.pkl', _path = p)
Xm_test = loadPickle('Xm_test.pkl', _path = p)
df_Stem = loadPickle('df_Stem.pkl', _path = p)
y_train = loadPickle('y_train.pkl', _path = p)
y_test = loadPickle('y_test.pkl', _path = p)

X_All = loadPickle('X_All.pkl', _path = p)
Y_All = loadPickle('y_All.pkl', _path = p)
X_Raw = loadPickle('X_Raw.pkl', _path = p)
X_Stem = loadPickle('X_Stem.pkl', _path = p)

gridResult = loadPickle('gridResult.pkl', _path = p)

gNameFeatures = loadPickle('gNameFeatures.pkl', _path = p)
gPredictionIdx = loadPickle('gPredictionIdx.pkl', _path = p)
GPredictionFeatures = loadPickle('gPredictionFeatures.pkl', _path = p)


#Gets dataframes of Predicted Feature values.

def getDFfromPickle(dataX):
    XPrediction = OrderedDict()
    gPredictionFeatures = OrderedDict()
    for k in kInstruments:
        #print '\n %s \n' % k.upper()        
        _A = dataX[k[0]][:]
        Xpreselected = [] 
        for i in gPredictionIdx[k]:

            Xpreselected.append(_A[:,i])

        XPrediction[k] = np.asarray(Xpreselected).T 

        x = []
        for f in GPredictionFeatures[k]:
            x.append('.'.join(f.split('.')[1::]))
        gPredictionFeatures[k] = x

    dfBass = pd.DataFrame(XPrediction[kInstruments[0]], columns = gPredictionFeatures[kInstruments[0]])
    dfGuitar = pd.DataFrame(XPrediction[kInstruments[1]], columns = gPredictionFeatures[kInstruments[1]])
    dfVocal = pd.DataFrame(XPrediction[kInstruments[2]], columns = gPredictionFeatures[kInstruments[2]])
    dfKeys = pd.DataFrame(XPrediction[kInstruments[3]], columns = gPredictionFeatures[kInstruments[3]])
    
    df = OrderedDict()
    df[kLabels[0]] = dfBass
    df[kLabels[1]] = dfGuitar
    df[kLabels[2]] = dfVocal
    df[kLabels[3]] = dfKeys
    
    return df

def getNPfromDF(df_R, df_S, trainSize = 0.9):
    
    A = OrderedDict()
    A_train = OrderedDict()
    A_test = OrderedDict() 
    As = OrderedDict()
    As_train = OrderedDict()
    As_test = OrderedDict() 
    Am = OrderedDict()
    Am_train = OrderedDict()
    Am_test = OrderedDict() 
    b = OrderedDict()
    b_train = OrderedDict()
    b_test = OrderedDict()
    
    
    
    
    for k in kLabels:
        
        df = df_Raw[k]
        df = df[np.abs(df-df.mean())<=(3*df.std())]
        idx = []
        for i in df.columns:
            a = pd.notnull(df[i])
            try:
                idx.append(np.where( a == False )[0][0])
                idx.append(np.where( a == False )[0][1])
            except:
                pass
        idx = list(set(idx))
        df.drop(df.index[idx], inplace=True)

        norm_df=(df-df.mean())/df.std()
        minmax_df=(df-df.min())/(df.max()-df.min())
        
        A[k] = pd.DataFrame.as_matrix(df)
        As[k] = pd.DataFrame.as_matrix(norm_df)
        Am[k] = pd.DataFrame.as_matrix(minmax_df)
        df2 = df_S[k]
        df2.drop(df2.index[idx], inplace=True)
        b[k] = pd.DataFrame.as_matrix(df2)
        A_train[k], A_test[k], As_train[k], As_test[k], Am_train[k], Am_test[k],b_train[k], b_test[k] = train_test_split(A[k],
                                                                                                                         As[k],
                                                                                                                         Am[k],
                                                                                                                         b[k],
                                                                                                                         train_size=trainSize,
                                                                                                                         random_state=42)
    return A_train, A_test, As_train, As_test, Am_train, Am_test, b_train, b_test
    


# define base model
def model_1(neurons=8, init_mode='zero', act_function='selu', dropout_rate=0.1, weight_constraint=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons*2, input_dim=4, 
                    kernel_initializer=init_mode, 
                    activation=act_function, 
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer=init_mode, activation=act_function,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, kernel_initializer=init_mode))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
# define base model

def model_2(neurons=8,init_mode='zero', act_function='selu', dropout_rate=0.1, weight_constraint=0):
    # create model
    model = Sequential()
    model.add(Dense(neurons*2, input_dim=6,
                    kernel_initializer=init_mode, 
                    activation=act_function, 
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer=init_mode, activation=act_function,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6, kernel_initializer=init_mode))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


#
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


label = sys.argv[1]
gridParams = sys.argv[2]
A_train = (X_train[label])
b_train = (y_train[label])
b_test = (y_test[label])
A_test = (X_test[label])


batch_size = [10, 20, 40, 50, 60, 70]
epochs = [50, 75, 100, 125, 150, 175, 200]
init_mode = ['zero','lecun_uniform']
activation = ['selu','sigmoid']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
neurons = [4, 8, 16, 32, 64]

if 'g' in label:

    est = KerasRegressor(build_fn=model_2, verbose=0)

else:

    est = KerasRegressor(build_fn=model_1, verbose=0)


if 'nbe' in gridParams:
    param_grid = dict(neurons=neurons,
                      batch_size=batch_size,
                      epochs=epochs)
elif 'neia' == gridParams:
    param_grid = dict(neurons=neurons,
                      epochs=epochs,
                      init_mode=init_mode,
                      act_function=activation)
elif 'ia' == gridParams:
    param_grid = dict(init_mode=init_mode,
                      act_function=activation)
elif 'dw' == gridParams:
    param_grid = dict(dropout_rate=dropout_rate,
                      weight_constraint=weight_constraint)




grid = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, verbose=0, cv=5, scoring = 'r2')
grid_result = grid.fit(A_train, b_train)



print("\n label - %s - Best: %f using %s" % (label, grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


gridResult[label+'_'+gridParams] = ("\n label - %s - Best: %f using %s" % (label, grid_result.best_score_, grid_result.best_params_))
gridResult[label+'_'+gridParams+'_means'] = means
gridResult[label+'_'+gridParams+'_params'] = params
gridResult[label+'_'+gridParams+'_stds'] = stds 
dumpPickle(gridResult, 'gridResult.pkl', _path = p)







