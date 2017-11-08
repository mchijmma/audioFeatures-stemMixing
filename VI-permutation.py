
# coding: utf-8

# In[1]:

from __future__ import division
from datetime import datetime


import numpy as np 
import pandas as pd


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
from sklearn.tree import DecisionTreeRegressor


from rf_perm_feat_import import PermutationImportance




#Constant Variables 
kGroup = sys.argv[1]
kInstrument = sys.argv[1]
kSampleRate = 44100
kN = 100
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
#Only if you initialized Variables.

X = loadPickle('X.pkl')
y = loadPickle('y.pkl')
XRaw = loadPickle('XRaw.pkl')
yRaw = loadPickle('yRaw.pkl')
XStem = loadPickle('XStem.pkl')
yStem = loadPickle('yStem.pkl')

gListTracks = loadPickle('gListTracks.pkl')
gNameFeatures = loadPickle('gNameFeatures.pkl')


gImportances = loadPickle('gImportances.pkl')
gPreSelectedImportances = loadPickle('gPreSelectedImportances.pkl')

gInterpretationOOBError = loadPickle('gInterpretationOOBError.pkl')
gInterpretationIdx = loadPickle('gInterpretationIdx.pkl')
gInterpretationFeatures = loadPickle('gInterpretationFeatures.pkl')

gPredictionOOBError = loadPickle('gPredictionOOBError.pkl')
gPredictionIdx = loadPickle('gPredictionIdx.pkl')
gPredictionOOBErrorGain = loadPickle('gPredictionOOBErrorGain.pkl')

gPreSelectedFeatures = loadPickle('gPreSelectedFeatures.pkl')
gPredictionFeatures = loadPickle('gPredictionFeatures.pkl')



    
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
  #gReducedFeatures[name] = gNameFeatures[name][:]
  
  
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
  #gReducedFeatures[name] = gReducedFeatures[list_instruments[0]][:]
  
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

#RAMDON FOREST FUNCTIONS:  

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


#Saves in global dict variables the results from the 3 iterations. 

def saveDictsinGVariables(data):
  
    X = loadPickle('X.pkl')
    y = loadPickle('y.pkl')
  
    X[kGroup.upper()] = data[:]
    y[kGroup.upper()] = y[kGroup]
    
    dumpPickle(X, 'X.pkl')
    dumpPickle(y, 'y.pkl')

    GImportances = loadPickle('gImportances.pkl')
    GPreSelectedImportances = loadPickle('gPreSelectedImportances.pkl')

    GInterpretationOOBError = loadPickle('gInterpretationOOBError.pkl')
    GInterpretationIdx = loadPickle('gInterpretationIdx.pkl')
    GInterpretationFeatures = loadPickle('gInterpretationFeatures.pkl')

    GPredictionOOBError = loadPickle('gPredictionOOBError.pkl')
    GPredictionIdx = loadPickle('gPredictionIdx.pkl')
    GPredictionOOBErrorGain = loadPickle('gPredictionOOBErrorGain.pkl')
    
    GPreSelectedFeatures = loadPickle('gPreSelectedFeatures.pkl')
    GPredictionFeatures = loadPickle('gPredictionFeatures.pkl')



    GImportances[kGroup] = gImportances[kGroup]
    GPreSelectedImportances[kGroup] = gPreSelectedImportances[kGroup]
    GImportances[kGroup.upper()] = gImportances[kGroup.upper()]
    GInterpretationOOBError[kGroup] = gInterpretationOOBError[kGroup]
    GInterpretationIdx[kGroup] = gInterpretationIdx[kGroup]
    GInterpretationFeatures[kGroup] = gInterpretationFeatures[kGroup]

    GPredictionOOBError[kGroup] = gPredictionOOBError[kGroup]
    GPredictionIdx[kGroup] = gPredictionIdx[kGroup]
    GPredictionOOBErrorGain[kGroup] = gPredictionOOBErrorGain[kGroup]
    
    GPreSelectedFeatures[kGroup] = gPreSelectedFeatures[kGroup]

    GPredictionFeatures[kGroup] = gPredictionFeatures[kGroup]


    dumpPickle(GPreSelectedFeatures, 'gPreSelectedFeatures.pkl')
    dumpPickle(GPredictionFeatures, 'gPredictionFeatures.pkl')

    dumpPickle(GImportances, 'gImportances.pkl')
    dumpPickle(GPreSelectedImportances, 'gPreSelectedImportances.pkl')

    dumpPickle(GInterpretationOOBError, 'gInterpretationOOBError.pkl')
    dumpPickle(GInterpretationIdx, 'gInterpretationIdx.pkl')
    dumpPickle(GInterpretationFeatures, 'gInterpretationFeatures.pkl')

    dumpPickle(GPredictionOOBError, 'gPredictionOOBError.pkl')
    dumpPickle(GPredictionIdx, 'gPredictionIdx.pkl')
    dumpPickle(GPredictionOOBErrorGain, 'gPredictionOOBErrorGain.pkl')
 


# Handy functions:
robust_scaler = RobustScaler()
min_max_scaler = preprocessing.MinMaxScaler()
min_abs_scaler = preprocessing.MaxAbsScaler()



_A = X[kGroup][:]
#_A = robust_scaler.fit_transform(_A)
label = y[kGroup]


startTime2 = datetime.now() 

  
startTime = datetime.now() 
nEstimators = 2000
_iterations = 20
epoch = 1
_type = 'Boot'
      
_indices = []
oob_error = []
importances = []
  
for i in range(_iterations):
    
    _epoch = epoch
    _importances = []
    _error = []
      
    if _type == 'No boot':      
      
        forest = RandomForestClassifier(n_estimators=nEstimators,
                                    warm_start=False,
                                    bootstrap = False,
                                    oob_score=False,
                                    max_features=0.3,
                                    random_state=None,
                                    n_jobs = -1)
    
        oobC = PermutationImportance()
    
#     forest2 = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
#                                           oob_score=True, max_features=0.3, random_state=None, n_jobs = -1)
                                                                    
        for j in range(_epoch):
                                          
            forest = forest.fit(_A, label)   
            feature_importances =  oobC.featureImportances(forest,
                                                           _A,
                                                           label,
                                                           10)
            print "Weighted Avg Information Gain feature importances:"
            print forest.feature_importances_
            print "Permutation importances:"
            print feature_importances
            importances.append(feature_importances)     

    elif _type == 'Boot':      
      
        forest = RandomForestClassifier(n_estimators=nEstimators,
                                    warm_start=False,
                                    bootstrap = True,
                                    oob_score=True,
                                    max_features=0.3,
                                    random_state=None,
                                    n_jobs = -1)
    
        oobC = PermutationImportance()
    
#     forest2 = RandomForestClassifier(n_estimators=nEstimators, warm_start=False, bootstrap = True,
#                                           oob_score=True, max_features=0.3, random_state=None, n_jobs = -1)
                                                                    
        for j in range(_epoch):
                                          
            forest = forest.fit(_A, label)   
            
            print "Weighted Avg Information Gain feature importances:"
            print forest.feature_importances_
            print "Permutation importances:"
            feature_importances =  oobC.featureImportances(forest,
                                                           _A,
                                                           label,
                                                           10)
            print feature_importances
            importances.append(feature_importances) 
          
          
            
    print '\n iteration # %d\n' % i


  
print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))
  





meanImportances = np.mean(importances, axis = 0)
stdImportances = np.std(importances, axis = 0)
kN = 500
gImportances[kGroup] = np.vstack((meanImportances, stdImportances))
indices = np.argsort(meanImportances)[::-1]
gImportances[kGroup.upper()] = importances
gImportances[kGroup] = gImportances[kGroup][:,indices]


#plt.plot(gImportances[kGroup][0][0:kN], 'k^', gImportances[kGroup][0][0:kN], 'k:',label=kGroup+'_mean' )
#plt.plot(gImportances[kGroup][1][0:kN], 'r^', gImportances[kGroup][1][0:kN], 'r:',label=kGroup+'_std' )



kN = 500

# Create a random dataset

X_stdTrain = np.arange(0,kN,1).reshape(-1,1)
y_stdTrain = gImportances[kGroup][1][0:kN][:].reshape(-1,1)


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=3)
regr_2 = DecisionTreeRegressor(max_depth=3)
regr_3 = DecisionTreeRegressor()
regr_1.fit(X_stdTrain, y_stdTrain)
regr_2.fit(X_stdTrain, y_stdTrain)
regr_3.fit(X_stdTrain, y_stdTrain)
# # Predict
X_test = np.arange(0.0, kN, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)



threshold = np.min(y_1[y_1>0.0001])
print '\n TH - STD - %f' % threshold
gPreSelectedImportances[kGroup] = gImportances[kGroup][0][gImportances[kGroup][0]>threshold]

#plt.plot(gPreSelectedImportances[kGroup], 'k^', gPreSelectedImportances[kGroup], 'k:',label=kGroup+'_pre_selected')
#plt.plot(X_test,y_1,'c')




idx = indices[0:len(gPreSelectedImportances[kGroup])]

gPreSelectedFeatures[kGroup] = getNameFeaturesIdx(idx, gNameFeatures[kGroup])



XPreSelected = []
for i in idx:
    
    XPreSelected.append(_A[:,i])
XPreSelected = np.asarray(XPreSelected).T  
  





startTime = datetime.now() 
  
oob_error = []
nEstimators = 2000
for i in range(len(idx)):
    
    _epoch = 20
    _error = [] 
    
      
    _A = XPreSelected[:,:i+1]
    print _A.shape
    
    forest = RandomForestClassifier(n_estimators=nEstimators,
                                    warm_start=False,
                                    bootstrap = True,
                                    oob_score=True,
                                    max_features=0.3,
                                    random_state=None,
                                    n_jobs = -1)
                                                                    
    for j in range(_epoch):
        
        forest = forest.fit(_A, label)                      
        _error.append(1 - forest.oob_score_)
        
    _error_mean = np.mean(_error, axis = 0) 
    oob_error.append(_error_mean)
      
    
    print 'Batch # %d - OOB error: %f \n' % (i, _error_mean) 
  
print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))


# In[65]:



gInterpretationOOBError[kGroup] = oob_error




print 'min_oob error at idx: %d\n' % np.argmin(oob_error)
gInterpretationFeatures[kGroup] = gPreSelectedFeatures[kGroup][:np.argmin(oob_error)+1]
print '\n Interpretation Features: %s' % gInterpretationFeatures[kGroup]

try:
    thresholdP = (1/(len(gPreSelectedFeatures[kGroup])-len(gInterpretationFeatures[kGroup]))) 
    _th = []
    for j in np.arange(len(gInterpretationFeatures[kGroup]),len(gPreSelectedFeatures[kGroup])-1,1):
        _th.append(np.abs(oob_error[j+1] - oob_error[j]))
    thresholdP = thresholdP * np.sum(_th)
except:
    thresholdP=0.001





print '\n TH - Error Gain - %f' % thresholdP



startTime = datetime.now() 
  
oob_error = []
oob_error_total = []
nEstimators = 2000
predIdx = idx.tolist()
errorGain = []
for i in range(len(idx)):
    
    _epoch = 20
    _error = [] 
    
      
    #_A = XPreSelected[:,:i+1]
    if i > 0:
        _A = np.hstack((_A,XPreSelected[:,i].reshape(-1,1)))
    else:
        _A = XPreSelected[:,:i+1]
                       
    print _A.shape
    
    forest = RandomForestClassifier(n_estimators=nEstimators,
                                    warm_start=False,
                                    bootstrap = True,
                                    oob_score=True,
                                    max_features=0.3,
                                    random_state=None,
                                    n_jobs = -1)
                                                                    
    for j in range(_epoch):
        
        forest = forest.fit(_A, label)                      
        _error.append(1 - forest.oob_score_)
    
    _error_mean = np.mean(_error, axis = 0) 
    oob_error.append(_error_mean)
    oob_error_total.append(_error_mean)
    if i > 0:
        _errorGain = oob_error[-2] - oob_error[-1]
        
        if _errorGain > thresholdP:
                
            print '\n Feature - %s, Idx - %d - Added' % (gPreSelectedFeatures[kGroup][i],i)
             
        else:
            _A = np.delete(_A, np.s_[-1], 1)
            predIdx.remove(idx[i])
            del oob_error[-1]
                
            print '\n Feature - %s, Idx - %d - Eliminated' % (gPreSelectedFeatures[kGroup][i],i)
            
            
                      
    else:
        
        _errorGain = 0
        print '\n Feature - %s, Idx - %d - Added' % (gPreSelectedFeatures[kGroup][i],i)
                
    #print _errorGain            
    errorGain.append(_errorGain)   

      
    
    print '\n Batch # %d - OOB error gain: %f \n' % (i, _errorGain) 
    
print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime))


# In[98]:

gPredictionFeatures[kGroup] = getNameFeaturesIdx(predIdx, gNameFeatures[kGroup])
gInterpretationIdx[kGroup] = idx.tolist()
gPredictionIdx[kGroup] = predIdx
gPredictionOOBError[kGroup] = oob_error_total
gPredictionOOBErrorGain[kGroup] = errorGain

print '\n Prediction Features: %s' % gPredictionFeatures[kGroup]



A = []
for i in predIdx:
    
    A.append(X[kGroup][:,i])
A = np.asarray(A).T  

saveDictsinGVariables(A)

print '\nExecuted in: %s. \n ' % (str(datetime.now() - startTime2))


