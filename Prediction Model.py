
# coding: utf-8

# In[ ]:


#This prediction model use a random forest prediction model to determine where conditions support growth. 
#Then, you will import the results it into ArcGIS Pro to find where the highest density of growth is likely to occur.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


import numpy as Num
import arcpy as ARCPY
import arcpy.da as DA
import pandas as PD
import seaborn as SEA
import matplotlib.pyplot as PLOT
import arcgisscripting as ARC
import SSUtilities as UTILS
import os as OS

#rename the datasets
inputFC = r'USA_Train'
globalFC = r'EMU_Global_90m_FillMissingVa'

#group the variables
predictVars = ['DISS02', 'NITRATE', 'PHOSPHATE','SALINITY','SILICATE', 'SRTM30', 'TEMP']
classVar = ['PRESENT']

#catconate the variable 
allVars = predictVars + classVar

#spaital reference the data 
trainFC = DA.FeatureClassToNumPyArray(inputFC, ["SHAPE@XY"] + allVars)
spatRef = ARCPY.Describe(inputFC).spatialReference

#create the structure for the training data 
data = PD.DataFrame(trainFC, columns = allVars)
corr = data.astype('float64').corr()

ax = SEA.heatmap(corr, cmap=SEA.diverging_palette(220, 10, as_cmap=True),
square=True, annot = True, linecolor = 'k', linewidths = 1)
PLOT.show()

#take the data sample size 
#fracNUM is the sample size of the data 
fracNum = 0.6
train_set = data.sample(frac = fracNum)
test_set = data.drop(train_set.index)
indicator, _ = PD.factorize(train_set[classVar[0]])
print('Training Data Size = ' + str(train_set.shape[0]))
print('Test Data Size = ' + str(test_set.shape[0]))



rfco = RandomForestClassifier(n_estimators = 500, oob_score = True)
rfco.fit(train_set[predictVars], indicator)
seagrassPred = rfco.predict(test_set[predictVars])
test_seagrass = test_set[classVar].as_matrix()
test_seagrass = test_seagrass.flatten()
error = NUM.sum(NUM.abs(test_seagrass - seagrassPred))/len(seagrassPred) * 100



test_seagrass = test_set[classVar].as_matrix()
test_seagrass = test_seagrass.flatten()
error = NUM.sum(NUM.abs(test_seagrass - seagrassPred))/len(seagrassPred) * 100
print('Accuracy = ' + str(100 - NUM.abs(error)) + '%')
print('locations with seagrass = ' + str(len(NUM.where(test_seagrass==1)[0])) )
print('Predicted Locations Seagrass = ' + str(len(NUM.where(seagrassPred==1)[0])))

#accuracy mean the % of times the model was correct out of 100
#the prediction model was correct in predicting seagrass occurrence in a location in which it was known to exist


rfco = RandomForestClassifier(n_estimators = 500)
rfco.fit(data[predictVars], indicatorUSA)
predictVars = ['DISSO2', 'NITRATE', 'PHOSPHATE', 'SALINITY', 'SILICATE', 'SRTM30', 'TEMP']
classVar = ['PRESENT']
allVars = predictVars + classVar
globalData = DA.FeatureClassToNumPyArray(globalFC, ["SHAPE@XY"] + predictVars)
spatRefGlobal = ARCPY.Describe(globalFC).spatialReference
globalTrain = PD.DataFrame(globalData, columns = predictVars)
seagrassPredGlobal = rfco.predict(globalTrain)
nameFC = 'GlobalPrediction2'
outputDir = r'C:\Users\ktpra\Documents\Seagrass\Seagrass.gdb'
grassExists = globalData[["SHAPE@XY"]][globalTrain.index[NUM.where(seagrassPredGlobal==1)]]
ARCPY.da.NumPyArrayToFeatureClass(grassExists, OS.path.join(outputDir, nameFC), ['SHAPE@XY'], spatRefGlobal)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
import numpy as Num
import arcpy as ARCPY
import arcpy.da as DA
import pandas as PD
import seaborn as SEA
import matplotlib.pyplot as PLOT
import arcgisscripting as ARC
import SSUtilities as UTILS
import os as OS

#rename the datasets
inputFC = r'USA_Train'
globalFC = r'EMU_Global_90m_FillMissingVa'

#group the variables
predictVars = ['DISS02', 'NITRATE', 'PHOSPHATE','SALINITY','SILICATE', 'SRTM30', 'TEMP']
classVar = ['PRESENT']

#catconate the variable 
allVars = predictVars + classVar

#spaital reference the data 
trainFC = DA.FeatureClassToNumPyArray(inputFC, ["SHAPE@XY"] + allVars)
spatRef = ARCPY.Describe(inputFC).spatialReference

#create the structure for the training data 
data = PD.DataFrame(trainFC, columns = allVars)
corr = data.astype('float64').corr()

ax = SEA.heatmap(corr, cmap=SEA.diverging_palette(220, 10, as_cmap=True),
square=True, annot = True, linecolor = 'k', linewidths = 1)
PLOT.show()

#take the data sample size 
#fracNUM is the sample size of the data 
fracNum = 0.6
train_set = data.sample(frac = fracNum)
test_set = data.drop(train_set.index)
indicator, _ = PD.factorize(train_set[classVar[0]])
print('Training Data Size = ' + str(train_set.shape[0]))
print('Test Data Size = ' + str(test_set.shape[0]))


import numpy as NUM
rfco = RandomForestClassifier(n_estimators = 500, oob_score = True)
rfco.fit(train_set[predictVars], indicator)
seagrassPred = rfco.predict(test_set[predictVars])
test_seagrass = test_set[classVar].as_matrix()
test_seagrass = test_seagrass.flatten()
error = NUM.sum(NUM.abs(test_seagrass - seagrassPred))/len(seagrassPred) * 100



test_seagrass = test_set[classVar].as_matrix()
test_seagrass = test_seagrass.flatten()
error = NUM.sum(NUM.abs(test_seagrass - seagrassPred))/len(seagrassPred) * 100
print('Accuracy = ' + str(100 - NUM.abs(error)) + '%')
print('locations with seagrass = ' + str(len(NUM.where(test_seagrass==1)[0])) )
print('Predicted Locations Seagrass = ' + str(len(NUM.where(seagrassPred==1)[0])))
indicatorUSA, _ = PD.factorize(data[classVar[0]])
rfco = RandomForestClassifier(n_estimators = 500)
rfco.fit(data[predictVars], indicatorUSA)
predictVars = ['DISSO2', 'NITRATE', 'PHOSPHATE', 'SALINITY', 'SILICATE', 'SRTM30', 'TEMP']
classVar = ['PRESENT']
allVars = predictVars + classVar
globalData = DA.FeatureClassToNumPyArray(globalFC, ["SHAPE@XY"] + predictVars)
spatRefGlobal = ARCPY.Describe(globalFC).spatialReference
globalTrain = PD.DataFrame(globalData, columns = predictVars)
seagrassPredGlobal = rfco.predict(globalTrain)
nameFC = 'GlobalPrediction2'
outputDir = r'C:\Users\ktpra\Documents\Seagrass\Seagrass.gdb'
grassExists = globalData[["SHAPE@XY"]][globalTrain.index[NUM.where(seagrassPredGlobal==1)]]
ARCPY.da.NumPyArrayToFeatureClass(grassExists, OS.path.join(outputDir, nameFC), ['SHAPE@XY'], spatRefGlobal)

