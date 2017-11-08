# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:49:33 2017

@author: marco
"""
from __future__ import division
from datetime import datetime


import numpy as np
import pandas as pd

from pylab import plot, show, figure, imshow
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')



import os
import pickle
import sys
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm


#%%
#Constant Variables
kGroup = sys.argv[1]
kInstrument = sys.argv[1]
#%%
kSampleRate = 44100
kN = 2048
kType = 'mono'
kFeatures = 1812 #Find a way to not hard-code this.
path = './Music/Data/MedleyDB/Features/%s/%s/' % (sys.argv[2], kType)

#Loads all global variables.

def dumpPickle(d, name):
  #path = './Music/Data/MedleyDB/Features/Arrays_original_1812/%s/' % (kType)

  with open(path + name, 'wb') as output:
    # Pickle dictionary using protocol 0.
    pickle.dump(d, output)

  print '%s Saved' % (name)

def loadPickle(name):

  #path = './Music/Data/MedleyDB/Features/Arrays_original_1812/%s/' % (kType)
  # load data from pkl file
  with open(path + name, "rb") as fp:
      loaded_data1 = pickle.load(fp)

  print '%s loaded, %s ' % (name, type(loaded_data1))

  return loaded_data1




#%%

# Definition of functions.



# Loads numpy array from Instrument name and path. It adds the to the global variables.
def loadsNumpyArray(name, _path = path):

  startTime = datetime.now()
  if not os.path.exists(_path):
      print 'WRONG PATH'

  XRaw[name] = np.load(_path + name + '_XRaw.npy')
  yRaw[name] = np.load(_path + name + '_yRaw.npy')
  XStem[name] = np.load(_path + name + '_XStem.npy')
  yStem[name] = np.load(_path + name + '_yStem.npy')

  X[name] = np.concatenate((XRaw[name], XStem[name]), axis = 0)
  y[name] = np.concatenate((yRaw[name], yStem[name]), axis = 0)

  gNameFeatures[name] = np.load(_path + name + '_gNameFeatures.npy').tolist()
  gListTracks[name] = np.load(_path + name + '_gListTracks.npy').tolist()
  gReducedFeatures[name] = gNameFeatures[name][:]


  print '\nExecuted in: %s. \n %d stems and raw %s tracks with %d features each were loaded.' % (str(datetime.now() - startTime), len(X[name]), name, len(gNameFeatures[name]))

# Create arrays for an specific group instruments. It adds them to the global variables.
def createGroupInstruments(name, list_instruments):

  _X = X[list_instruments[0]][:]
  _Xr = XRaw[list_instruments[0]][:]
  _Xs = XStem[list_instruments[0]][:]

  _y = y[list_instruments[0]][:]
  _yr = yRaw[list_instruments[0]][:]
  _ys = yStem[list_instruments[0]][:]

  _gListTracks = gListTracks[list_instruments[0]][:]
  gNameFeatures[name] = gNameFeatures[list_instruments[0]][:]
  gReducedFeatures[name] = gReducedFeatures[list_instruments[0]][:]

  del list_instruments[0]

  for inst in list_instruments:

      _X = np.concatenate((_X, X[inst]), axis = 0)
      _y = np.concatenate((_y, y[inst]), axis = 0)
      _gListTracks = _gListTracks + gListTracks[inst]


  X[name] = _X
  y[name] = _y
  XRaw[name] = _Xr
  yRaw[name] = _yr
  XStem[name] = _Xs
  yStem[name] = _ys

  gListTracks[name] = _gListTracks

# RAMDON FOREST FUNCTIONS:

# Check OOB ERROR against number of trees.

def getNumberTrees(_X, _y, min_estimators = 10, max_estimators = 1000, epoch = 10):

    startTime = datetime.now()


    forest = RandomForestClassifier(n_estimators=10, warm_start=True, bootstrap = True,
                                    oob_score=True,
                                    max_features="sqrt", random_state=None, n_jobs = -1)

    oob_error_ntrees = []

    for i in range(min_estimators, max_estimators + 1):

      for j in range(epoch):
        oob = []
        forest.set_params(n_estimators=i)
        forest.fit(_X, _y)

            # Record the OOB error for each `n_estimators=i` setting.
        _oob_error = 1 - forest.oob_score_
        oob.append(_oob_error)

      oob_error_ntrees.append((i, np.mean(oob)))

    print '\nExecuted in: %s. \n' % (str(datetime.now() - startTime))
    # Generate the "OOB error rate" vs. "n_estimators" plot.

    #plotOOBError(oob_error_ntrees, xlabel = 'n_estimators - %s' % kGroup)
    n = np.argmin((np.asarray(oob_error_ntrees))[:,1]) + min_estimators

    return oob_error_ntrees, n

# Plots OOB error against different type of iterations.
def plotOOBError(error, xlabel = 'n_features', ylabel = 'OOB error rate'):
    _x = []
    _y = []

    for tupla in error:
        _x.append(tupla[0])
        _y.append(tupla[1])
    plt.rcParams['figure.figsize'] = (18,15)
    plt.plot(_x, _y, 'k*')

    plt.xlim(np.min(_x), np.max(_x)+100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.show()

def getNameFeaturesIdx(idx, nameFeatures):#, NameFeatures = gNameFeatures):
  a = []

  for i in idx:

      _a  = (nameFeatures[i])
      print _a
      a.append(_a)

  return a


# Random Forest Classifier that does N epochs, and calculates mean OOB error and mean Importances indices

def RFmeanOOBerror(data, label, nameFeatures, epoch = 100, nEstimators = 100,
                   nFeatures = 10):

  # iterates epoch time to have mean OOB and mean relevance of features.
  startTime = datetime.now()

  _importances = []
  _error = []
  forest = RandomForestClassifier(n_estimators=nEstimators, warm_start=True, bootstrap = True,
                                  oob_score=True,
                                  max_features='sqrt', random_state=None, n_jobs = -1)
  for i in range(epoch):


      forest = forest.fit(data, label)
      _importances.append(forest.feature_importances_)
      _error.append(1 - forest.oob_score_)

  _error_mean = np.mean(_error, axis = 0)
  _importances = np.mean(_importances, axis = 0)
  _indices = np.argsort(_importances)[::-1]
  print '\n Indices : \n'
  print _indices
  print '\n %d Relevant Features: \n ' % nFeatures
  getNameFeaturesIdx(_indices[0:nFeatures], nameFeatures)
  print '\n OOB error: %f \n' % (_error_mean)

  print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))


# Gives approximate OOB error usign different types of scalling
def RFmeanScallingError(data, label, nEstimators = 100, epoch = 10):


    A = data[:]
    X_rs = robust_scaler.fit_transform(A)
    X_n = preprocessing.normalize(A)
    X_mm = min_max_scaler.fit_transform(A)
    X_ma = min_abs_scaler.fit_transform(A)
    X_mars = min_abs_scaler.fit_transform(X_rs)


    _error = []

    for n,i in enumerate([A,X_rs, X_n, X_mm, X_ma, X_mars]):

        forest = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
                                    oob_score=True, max_features="sqrt",
                                    random_state=None, n_jobs = -1)

        for j in range(epoch):

            forest.fit(i, label)

            # Record the OOB error for each `n_estimators=i` setting.
            _error.append(1 - forest.oob_score_)

        print n, np.mean(_error)
    print '\n 0 - data\n 1 - robust scaler\n 2 - normalize\n 3 - min-max scaler\n 4 - min-abs scaler\n 5 - min-abs-robust scaler'


# Starts removing the last feautre in importance after N epoch iterations.
# If itera = 0, it will remove until the last feature.
# It returns a dictionary with Deleted, Reduced, OOb error, sorted Features, and remaining X values.
# It uses intern dicts because I'm too lazy, but it works.

def removeFeatures(data, label, features, itera = 0, epoch = 10, nEstimators = 128):

  _A = data[:]
  startTime = datetime.now()
  gDeletedFeatures2 = OrderedDict()
  gReducedFeatures2 = OrderedDict()
  gReducedFeatures2[kInstrument] = features[:]

  startTime = datetime.now()
  if itera == 0:
      _iterations = _A.shape[1] - 1#Always do it with i = n-1 if 100, do 99.
  else:
      _iterations = itera

  _indices = []
  oob_error = []


  for i in range(_iterations):

      _epoch = epoch
      _importances = []
      _error = []

      forest = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = False,
                                      oob_score=False, max_features='sqrt', random_state=None, n_jobs = -1)

      forest2 = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
                                      oob_score=True, max_features='sqrt', random_state=None, n_jobs = -1)

      for j in range(_epoch):



          forest = forest.fit(_A, label)
          forest2 = forest2.fit(_A, label)

          _importances.append(forest.feature_importances_)
          _error.append(1 - forest2.oob_score_)



      #Gets index of last feature
      importances = np.mean(_importances, axis = 0)
      indices = np.argsort(importances)[::-1]
      #save deleted idx
      _indices.append(indices[-1])


      _A = np.delete(_A, np.s_[indices[-1]], 1)
      gDeletedFeatures2.setdefault(kInstrument, []).append(gReducedFeatures2[kInstrument][indices[-1]])
      del gReducedFeatures2[kInstrument][indices[-1]]


      _error_mean = np.mean(_error, axis = 0)
      oob_error.append((i, _error_mean))

      #prints every 32 iterations (31) or 11
      if i % int(np.ceil(_iterations*0.1)) == 0:
          print 'OOB error: %f \n' % (_error_mean)


  # Generate the "OOB error rate" vs. "n_features" plot.

  #plotOOBError(oob_error, xlabel = 'n_features_removed - %s' % kGroup)

  _features = gReducedFeatures2[kInstrument] + list(reversed(gDeletedFeatures2[kInstrument]))

  result = OrderedDict()
  result['Deleted'] = gDeletedFeatures2[kInstrument]
  result['Reduced'] = gReducedFeatures2[kInstrument]
  result['OOB error'] = oob_error
  result['Features'] = _features
  result['X'] = _A


  print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))

  return result






# Returns the  array containing selected features (from deleted and reduced),
# number = 0 means it will return from the minimum oob_error. Returns pd Dataframe as well.

def getArraysOfRelevantFeatures(error,
                                deleted,
                                reduced,
                                number = 0, type = 1):

  _X = X[kGroup][:]
  _y = y[kGroup]

  __X = []

  if type == 2:
      error = gOOBError[kGroup+'2']
      deleted = gDeletedFeatures[kGroup+'2']
      reduced = gReducedFeatures[kGroup+'2']

  elif type == 3:
      error = gOOBError[kGroup+'3']
      deleted = gDeletedFeatures[kGroup+'3']
      reduced = gReducedFeatures[kGroup+'3']

  elif type == 4:
      error = gOOBError[kGroup+'4']
      deleted = gDeletedFeatures[kGroup+'4']
      reduced = gReducedFeatures[kGroup+'4']




  if number == 0:
      idx = np.argmin(np.asarray(error)[:,1])
  else:
      idx = -1*(number - 1)
  features = deleted[idx::]
  features.append(reduced[0])
  features = list(reversed(features))
  idx = []

  for name in features:

      idx.append(gNameFeatures[kGroup].index(name))

  for i in idx:

      __X.append(_X[:,i])

  __X = np.asarray(__X).T

  df = pd.DataFrame(__X, columns = features)

  df = df.assign(stem = _y)

  return __X, features, df

# Performs 4 iterations of feature removal, returns final array, final selected features,
# dataframe, and three dicts that were created in the process.

def removeFeatures3(data, label, nEstimators = 100):
  # Start removing the last important feature after a certain number of iterations.
  # Gets dict with Deleted, Reduced, sorted Features and OOB error.
  print 'dict1'
  data = robust_scaler.fit_transform(data)
  oob_error_ntrees, n = getNumberTrees(data, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 10)
  print '# trees: %d' % n
  dict1 = removeFeatures(data, label, gReducedFeatures[kGroup],
                                 itera = kFeatures-101, epoch = 2, nEstimators = n)
  print 'dict2'
  dict2 = removeFeatures(dict1['X'], label, dict1['Reduced'],
                             itera = 0, epoch = 10, nEstimators = n)

  data2, dfeatures, dfGuitar = getArraysOfRelevantFeatures(error = dict2['OOB error'],
                                                    deleted = dict2['Deleted'],
                                                    reduced = dict2['Reduced'],
                                                    number = 20, type = 1)
  # if you decide to go ahead with an scalling.
  data2 = robust_scaler.fit_transform(data2)
  #data2 = preprocessing.normalize(data2)
  #data2 = min_max_scaler.fit_transform(data2)
  #data2 = min_abs_scaler.fit_transform(data2)
  #data2 = min_abs_scaler.fit_transform(robust_scaler.fit_transform(data2))

  oob_error_ntrees, n = getNumberTrees(data2, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 10)
  print '# trees: %d' % n
  print 'dict3'
  dict3 = removeFeatures(data2, label, dfeatures,
                                  itera = 0, epoch = 100, nEstimators = n)

  data3, dfeatures, df = getArraysOfRelevantFeatures(error = dict3['OOB error'],
                                                    deleted = dict3['Deleted'],
                                                    reduced = dict3['Reduced'],
                                                    number = 0, type = 1)
  data3 = robust_scaler.fit_transform(data3)

  oob_error_ntrees, n = getNumberTrees(data3, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 50)
  print '# trees: %d' % n
  print 'dict4'
  dict4 = removeFeatures(data3, label, dfeatures,
                                  itera = 0, epoch = 1000, nEstimators = n)

  data4, dfeatures, df = getArraysOfRelevantFeatures(error = dict4['OOB error'],
                                                    deleted = dict4['Deleted'],
                                                    reduced = dict4['Reduced'],
                                                    number = 0, type = 1)

  return data4, dfeatures, df, dict1, dict2, dict3, dict4, n

# Performs 4 iterations of feature removal, returns final array, final selected features,
# dataframe, and three dicts that were created in the process.

def removeFeatures4Permutation(data, label, nEstimators = 100):
  # Start removing the last important feature after a certain number of iterations.
  # Gets dict with Deleted, Reduced, sorted Features and OOB error.
  print 'dict1'
  data = robust_scaler.fit_transform(data)
  oob_error_ntrees, n = getNumberTrees(data, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 10)
  print '# trees: %d' % n
  dict1 = removeFeaturesPermutation(data, label, gReducedFeatures[kGroup],
                                 itera = kFeatures-101, epoch = 2, nEstimators = n)
  print 'dict2'
  dict2 = removeFeaturesPermutation(dict1['X'], label, dict1['Reduced'],
                             itera = 0, epoch = 10, nEstimators = n)

  data2, dfeatures, dfGuitar = getArraysOfRelevantFeatures(error = dict2['OOB error'],
                                                    deleted = dict2['Deleted'],
                                                    reduced = dict2['Reduced'],
                                                    number = 20, type = 1)
  # if you decide to go ahead with an scalling.
  data2 = robust_scaler.fit_transform(data2)
  #data2 = preprocessing.normalize(data2)
  #data2 = min_max_scaler.fit_transform(data2)
  #data2 = min_abs_scaler.fit_transform(data2)
  #data2 = min_abs_scaler.fit_transform(robust_scaler.fit_transform(data2))

  oob_error_ntrees, n = getNumberTrees(data2, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 10)
  print '# trees: %d' % n
  print 'dict3'
  dict3 = removeFeaturesPermutation(data2, label, dfeatures,
                                  itera = 0, epoch = 100, nEstimators = n)

  data3, dfeatures, df = getArraysOfRelevantFeatures(error = dict3['OOB error'],
                                                    deleted = dict3['Deleted'],
                                                    reduced = dict3['Reduced'],
                                                    number = 0, type = 1)
  data3 = robust_scaler.fit_transform(data3)

  oob_error_ntrees, n = getNumberTrees(data3, label, min_estimators = 50,
                                  max_estimators = 200, epoch = 50)
  print '# trees: %d' % n
  print 'dict4'
  dict4 = removeFeaturesPermutation(data3, label, dfeatures,
                                  itera = 0, epoch = 100, nEstimators = n)

  data4, dfeatures, df = getArraysOfRelevantFeatures(error = dict4['OOB error'],
                                                    deleted = dict4['Deleted'],
                                                    reduced = dict4['Reduced'],
                                                    number = 0, type = 1)

  return data4, dfeatures, df, dict1, dict2, dict3, dict4, n


#Saves in global dict variables the results from the 3 iterations.

def saveDictsinGVariables(dict1, dict2, dict3, dict4, n, df, A, features):





  gDeletedFeatures = loadPickle('gDeletedFeatures.pkl')
  gOOBError = loadPickle('gOOBError.pkl')
  gReducedFeatures = loadPickle('gReducedFeatures.pkl')
  gNTrees = loadPickle('gNTrees.pkl')
  gFeatures = loadPickle('gFeatures.pkl')
  X = loadPickle('X.pkl')
  y = loadPickle('y.pkl')

  X[kGroup.upper()] = A[:]
  y[kGroup.upper()] = y[kGroup]

  gOOBError[kGroup] = dict1['OOB error']
  gDeletedFeatures[kGroup] = dict1['Deleted']
  gReducedFeatures[kGroup] = dict1['Reduced']

  gNTrees[kGroup] = n
  gFeatures[kGroup] = features

  gOOBError[kGroup+'2'] = dict2['OOB error']
  gDeletedFeatures[kGroup+'2'] = dict2['Deleted']
  gReducedFeatures[kGroup+'2'] = dict2['Reduced']

  gOOBError[kGroup+'3'] = dict3['OOB error']
  gDeletedFeatures[kGroup+'3'] = dict3['Deleted']
  gReducedFeatures[kGroup+'3'] = dict3['Reduced']

  gOOBError[kGroup+'4'] = dict4['OOB error']
  gDeletedFeatures[kGroup+'4'] = dict4['Deleted']
  gReducedFeatures[kGroup+'4'] = dict4['Reduced']

  dumpPickle(dict1, 'd_%s.pkl' % kGroup)
  dumpPickle(dict2, 'd_%s2.pkl' % kGroup)
  dumpPickle(dict3, 'd_%s3.pkl' % kGroup)
  dumpPickle(dict4, 'd_%s4.pkl' % kGroup)
  dumpPickle(df, 'df%s.pkl' % kGroup)

  dumpPickle(X, 'X.pkl')
  dumpPickle(y, 'y.pkl')

  dumpPickle(gDeletedFeatures, 'gDeletedFeatures.pkl')
  dumpPickle(gOOBError, 'gOOBError.pkl')
  dumpPickle(gReducedFeatures, 'gReducedFeatures.pkl')
  dumpPickle(gNTrees, 'gNTrees.pkl')
  dumpPickle(gFeatures, 'gFeatures.pkl')

# Handy functions:
robust_scaler = RobustScaler()
min_max_scaler = preprocessing.MinMaxScaler()
min_abs_scaler = preprocessing.MaxAbsScaler()

from rf_perm_feat_import import PermutationImportance


def removeFeaturesPermutation(data, label, features, itera = 0, epoch = 10, nEstimators = 128):

  _A = data[:]
  startTime = datetime.now()
  gDeletedFeatures2 = OrderedDict()
  gReducedFeatures2 = OrderedDict()
  gReducedFeatures2[kInstrument] = features[:]

  startTime = datetime.now()
  if itera == 0:
      _iterations = _A.shape[1] - 1#Always do it with i = n-1 if 100, do 99.
  else:
      _iterations = itera

  _indices = []
  oob_error = []


  for i in range(_iterations):

      _epoch = epoch
      _importances = []
      _error = []

      forest = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
                                      oob_score=True, max_features='sqrt', random_state=None, n_jobs = -1)

#      forest2 = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
#                                      oob_score=True, max_features='sqrt', random_state=None)


      for j in range(_epoch):

          forest = forest.fit(_A, label)


          oobC = PermutationImportance()



          feature_importances =  oobC.featureImportances(forest, _A, label, _epoch)

          _importances.append(feature_importances)
          _error.append(1 - forest.oob_score_)



      #Gets index of last feature
      importances = np.mean(_importances, axis = 0)
      indices = np.argsort(importances)[::-1]
      #save deleted idx
      _indices.append(indices[-1])


      _A = np.delete(_A, np.s_[indices[-1]], 1)
      gDeletedFeatures2.setdefault(kInstrument, []).append(gReducedFeatures2[kInstrument][indices[-1]])
      del gReducedFeatures2[kInstrument][indices[-1]]


      _error_mean = np.mean(_error, axis = 0)
      oob_error.append((i, _error_mean))

      #prints every 32 iterations (31) or 11
      if i % int(np.ceil(_iterations*0.01)) == 0:
          print 'OOB error: %f \n' % (_error_mean)


  # Generate the "OOB error rate" vs. "n_features" plot.

  #plotOOBError(oob_error, xlabel = 'n_features_removed - %s' % kGroup)

  _features = gReducedFeatures2[kInstrument] + list(reversed(gDeletedFeatures2[kInstrument]))

  result = OrderedDict()
  result['Deleted'] = gDeletedFeatures2[kInstrument]
  result['Reduced'] = gReducedFeatures2[kInstrument]
  result['OOB error'] = oob_error
  result['Features'] = _features
  result['X'] = _A


  print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))

  return result

#%%
# Start Random Forest Classification
# This is done only the first time.
kSampleRate = 44100
kN = 2048
kType = 'mono'
kFeatures = 1812 
path = './Music/Data/MedleyDB/Features/Arrays_original_1812_nw_permutation/%s/' % (kType)
X = OrderedDict()
y = OrderedDict()
XRaw = OrderedDict()
yRaw = OrderedDict()
XStem = OrderedDict()
yStem = OrderedDict()
gNameFeatures = OrderedDict()
gListTracks = OrderedDict()
gReducedFeatures = OrderedDict()
gDeletedFeatures = OrderedDict()
gOOBError = OrderedDict()
gNTrees = OrderedDict()
gFeatures = OrderedDict()

# Loads numpy arrays
loadsNumpyArray('electric bass')
loadsNumpyArray('piano')
loadsNumpyArray('tack piano')
loadsNumpyArray('electric piano')
loadsNumpyArray('synthesizer')
loadsNumpyArray('acoustic guitar')
loadsNumpyArray('clean electric guitar')
loadsNumpyArray('distorted electric guitar')
loadsNumpyArray('banjo')
loadsNumpyArray('male singer')
loadsNumpyArray('female singer')
loadsNumpyArray('male rapper')
loadsNumpyArray('vocalists')
##%%

# Create 'bass' group
kGroup = 'bass1'
createGroupInstruments(kGroup,['electric bass'])
##%%
# Create 'key' group
kGroup = 'keys1'
createGroupInstruments(kGroup,['piano',
                              'electric piano'])
##%%
# Create 'key' group
kGroup = 'keys-tack1'
createGroupInstruments(kGroup,['piano',
                              'tack piano',
                              'electric piano'])
##%%
# Create 'key-synth' group
kGroup = 'keys-synth1'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'synthesizer'])
                              # Create 'key-synth' group
kGroup = 'keys-all1'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'tack piano',
                              'synthesizer'])
##%%
# Create 'vocal' group
kGroup = 'vocal1'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'male rapper'])
##%%
# Create 'vocal' group
kGroup = 'vocal-vocalists1'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'vocalists',
                              'male rapper'])
##%%
# Create 'guitar' group
kGroup = 'guitar1'
createGroupInstruments(kGroup,['acoustic guitar',
                              'clean electric guitar',
                              'distorted electric guitar',
                              'banjo'])

# Create 'bass' group
kGroup = 'bass2'
createGroupInstruments(kGroup,['electric bass'])
##%%
# Create 'key' group
kGroup = 'keys2'
createGroupInstruments(kGroup,['piano',
                              'electric piano'])
##%%
# Create 'key' group
kGroup = 'keys-tack2'
createGroupInstruments(kGroup,['piano',
                              'tack piano',
                              'electric piano'])
##%%
# Create 'key-synth' group
kGroup = 'keys-synth2'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'synthesizer'])
                              # Create 'key-synth' group
kGroup = 'keys-all2'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'tack piano',
                              'synthesizer'])
##%%
# Create 'vocal' group
kGroup = 'vocal2'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'male rapper'])
##%%
# Create 'vocal' group
kGroup = 'vocal-vocalists2'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'vocalists',
                              'male rapper'])
##%%
# Create 'guitar' group
kGroup = 'guitar2'
createGroupInstruments(kGroup,['acoustic guitar',
                              'clean electric guitar',
                              'distorted electric guitar',
                              'banjo'])

# Create 'bass' group
kGroup = 'bass3'
createGroupInstruments(kGroup,['electric bass'])
##%%
# Create 'key' group
kGroup = 'keys3'
createGroupInstruments(kGroup,['piano',
                              'electric piano'])
##%%
# Create 'key' group
kGroup = 'keys-tack3'
createGroupInstruments(kGroup,['piano',
                              'tack piano',
                              'electric piano'])
##%%
# Create 'key-synth' group
kGroup = 'keys-synth3'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'synthesizer'])
                              # Create 'key-synth' group
kGroup = 'keys-all3'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'tack piano',
                              'synthesizer'])
##%%
# Create 'vocal' group
kGroup = 'vocal3'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'male rapper'])
##%%
# Create 'vocal' group
kGroup = 'vocal-vocalists3'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'vocalists',
                              'male rapper'])
##%%
# Create 'guitar' group
kGroup = 'guitar3'
createGroupInstruments(kGroup,['acoustic guitar',
                              'clean electric guitar',
                              'distorted electric guitar',
                              'banjo'])

# Create 'bass' group
kGroup = 'bass4'
createGroupInstruments(kGroup,['electric bass'])
##%%
# Create 'key' group
kGroup = 'keys4'
createGroupInstruments(kGroup,['piano',
                              'electric piano'])
##%%
# Create 'key' group
kGroup = 'keys-tack4'
createGroupInstruments(kGroup,['piano',
                              'tack piano',
                              'electric piano'])
##%%
# Create 'key-synth' group
kGroup = 'keys-synth4'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'synthesizer'])
                              # Create 'key-synth' group
kGroup = 'keys-all4'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'tack piano',
                              'synthesizer'])
##%%
# Create 'vocal' group
kGroup = 'vocal4'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'male rapper'])
##%%
# Create 'vocal' group
kGroup = 'vocal-vocalists4'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'vocalists',
                              'male rapper'])
##%%
# Create 'guitar' group
kGroup = 'guitar4'
createGroupInstruments(kGroup,['acoustic guitar',
                              'clean electric guitar',
                              'distorted electric guitar',
                              'banjo'])

# Create 'bass' group
kGroup = 'bass5'
createGroupInstruments(kGroup,['electric bass'])
##%%
# Create 'key' group
kGroup = 'keys5'
createGroupInstruments(kGroup,['piano',
                              'electric piano'])
##%%
# Create 'key' group
kGroup = 'keys-tack5'
createGroupInstruments(kGroup,['piano',
                              'tack piano',
                              'electric piano'])
##%%
# Create 'key-synth' group
kGroup = 'keys-synth5'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'synthesizer'])
                              # Create 'key-synth' group
kGroup = 'keys-all5'
createGroupInstruments(kGroup,['piano',
                              'electric piano',
                              'tack piano',
                              'synthesizer'])
##%%
# Create 'vocal' group
kGroup = 'vocal5'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'male rapper'])
##%%
# Create 'vocal' group
kGroup = 'vocal-vocalists5'
createGroupInstruments(kGroup,['male singer',
                              'female singer',
                              'vocalists',
                              'male rapper'])
##%%
# Create 'guitar' group
kGroup = 'guitar5'
createGroupInstruments(kGroup,['acoustic guitar',
                              'clean electric guitar',
                              'distorted electric guitar',
                              'banjo'])

##%%
# SAVE !

#
dumpPickle(X, 'X.pkl')
dumpPickle(y, 'y.pkl')
dumpPickle(XRaw, 'XRaw.pkl')
dumpPickle(yRaw, 'yRaw.pkl')
dumpPickle(XStem, 'XStem.pkl')
dumpPickle(yStem, 'yStem.pkl')
dumpPickle(gDeletedFeatures, 'gDeletedFeatures.pkl')
dumpPickle(gListTracks, 'gListTracks.pkl')
dumpPickle(gNameFeatures, 'gNameFeatures.pkl')
dumpPickle(gOOBError, 'gOOBError.pkl')
dumpPickle(gReducedFeatures, 'gReducedFeatures.pkl')
dumpPickle(gNTrees, 'gNTrees.pkl')
dumpPickle(gFeatures, 'gFeatures.pkl')
