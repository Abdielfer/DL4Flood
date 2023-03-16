import os
import glob
import pathlib
import shutil
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 

### General applications ##
class timeit(): 
    '''
    to compute execution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name

## modeling manipulation
def saveModel(estimator, id):
    name = id + ".pkl" 
    _ = joblib.dump(estimator, name, compress=9)

def loadModel(modelName):
    return joblib.load(modelName)

def logTransformation(x):
    '''
    Logarithmic transformation to redistribute values between 0 and 1. 
    '''
    x_nonZeros = np.where(x <= 0.0000001, 0.0001, x)
    return np.max(np.log(x_nonZeros)**2) - np.log(x_nonZeros)**2

def createWeightVector(y_vector, dominantValue:float, dominantValuePenalty:float):
    '''
    Create wight vector for sampling weighted training.
    The goal is to penalize the dominant class. 
    This is important is the flood study, where majority of points (usually more than 95%) 
    are not flooded areas. 
    '''
    y_ravel  = (np.array(y_vector).astype('int')).ravel()
    weightVec = np.ones_like(y_ravel).astype(float)
    weightVec = [dominantValuePenalty if y_ravel[j] == dominantValue else 1 for j in range(len(y_ravel))]
    return weightVec

####  Sampling manipulation
def stratifiedSampling(dataSetName, targetColName):
    '''
    Performe a sampling that preserve classses proportions on both, train and test sets.
    '''
    X,Y = importDataSet(dataSetName, targetColName)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=50)
    for train_index, test_index in sss.split(X, Y):
        print("TRAIN:", train_index.size, "TEST:", test_index)
        X_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
    
    return X_train, y_train, X_test, y_test

def randomUndersampling(x_DataSet, y_DataSet, sampling_strategy = 'auto'):
    sm = RandomUnderSampler(random_state=50, sampling_strategy=sampling_strategy)
    x_res, y_res = sm.fit_resample(x_DataSet, y_DataSet)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res

def removeCoordinatesFromDataSet(dataSet):
    '''
    Remove colums of coordinates if they exist in dataset
    @input:
      @dataSet: pandas dataSet
    '''
    DSNoCoord = dataSet
    if 'x_coord' in DSNoCoord.keys(): 
      DSNoCoord.drop(['x_coord','y_coord'], axis=1, inplace = True)
    else: 
      print("DataSet has no coordinates to remove")
    return DSNoCoord

### Modifying class domain
def pseudoClassCreation(dataset, conditionVariable, threshold, pseudoClass, targetColumnName):
    '''
    Replace <targetClass> by  <pseudoClass> where <conditionVariable >= threshold>. 
    Return:
      dataset with new classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    conditionVar = (np.array(dataset[conditionVariable])).ravel()
    datasetReclassified[targetColumnName] = [ pseudoClass if conditionVar[j] >= threshold 
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def revertPseudoClassCreation(dataset, originalClass, pseudoClass, targetColumnName):
    '''
    Restablich  <targetClass> with <originalClass> where <targetColumnName == pseudoClass>. 
    Return:
      dataset with original classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    datasetReclassified[targetColumnName] = [ originalClass if actualTarget[j] == pseudoClass
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def makeBinary(dataset,targetColumn,classToKeep:int, replacerClassName:int):
    '''
    makeBinary(dataset,targetColumn,classToKeep, replacerClassName)
    @classToKeep @input: Class name to conserv. All different classes will be repleced by <replacerClassName>
    '''
    repalcer  = dataset[targetColumn].to_numpy()
    dataset[targetColumn] = [replacerClassName if repalcer[j] != classToKeep else repalcer[j] for j in range(len(repalcer))]  
    return dataset

### Configurations And file management
def importConfig():
    with open('./config.txt') as f:
        content = f.readlines()
    print(content)    
    return content

def getLocalPath():
    return os.getcwd()

def makePath(str1,str2):
    return os.path.join(str1,str2)

def ensureDirectory(pathToCheck):
    if os.path.isdir(pathToCheck): 
        return pathToCheck
    else:
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
        return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath ust contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

def createTransitFolder(parent_dir_path):
    path = os.path.join(parent_dir_path, 'TransitDir')
    ensureDirectory(path)
    return path

def clearTransitFolderContent(path, filetype = '/*'):
    '''
    NOTE: This well clear dir without removing the parent dir itself. 
    We can replace '*' for an specific condition ei. '.tif' for specific fileType deletion if needed. 
    @Arguments:
    @path: Parent directory path
    @filetype: file type toi delete. @default ='/*' delete all files. 
    '''
    files = glob.glob(path + filetype)
    for f in files:
        os.remove(f)
    return True

def listFreeFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    for (root, dirs, file) in os.walk(cwd):
        file_list = [i for i in file if ext in i]
        return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    FILE_LIST = []
    for (root, dirs, file) in os.walk(cwd):
        for i in file:
            if ext in i:
                FILE_LIST.append(i)
    return FILE_LIST

def createListFromCSVColumn(csv_file_location, col_id:str):  
    '''
    @return: list from <col_id> in <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.read_csv(csv_file_location, index_col = None)
    for i in df[col_id]:
        x.append(i)
    return x

def createListFromExelColumn(excell_file_location,Sheet_id:str, col_id:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_id]:
        x.append(i)
    return x

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 

def importDataSet(dataSetName, targetCol: str):
    '''
    Import datasets and return         
    @input: DataSetName => The dataset path. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(dataSetName, index_col = None)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    return x, y


 ### Metrics ####  
def accuracyFromConfusionMatrix(confusion_matrix):
    '''
    Only for binary
    '''
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def pritnAccuracy(y_predic, y_val):
    '''
    Only for binary
    '''
    cm = confusion_matrix(y_predic, y_val) 
    print("Accuracy of MLPClassifier : ", accuracyFromConfusionMatrix(cm)) 

            ###########            
            ### GIS ###
            ###########

def makePredictionToImportAsSHP(model, x_test, y_test, targetColName):
    '''
    We asume here that x_test contain coordinates as <x_coord> and <y_coord>.
    Return:
         The full dataset including a prediction column.  
    '''
    x_testCopy = x_test 
    sampleCoordinates = pd.DataFrame()
    sampleCoordinates['x_coord'] = x_testCopy['x_coord']
    sampleCoordinates['y_coord'] = x_testCopy['y_coord']
    x_testCopy.drop(['x_coord','y_coord'], axis=1, inplace = True)
    y_hay = model.predict(x_testCopy)
    ds_toSHP = x_testCopy
    ds_toSHP[targetColName] = y_test
    ds_toSHP['x_coord'] = sampleCoordinates['x_coord']
    ds_toSHP['y_coord'] = sampleCoordinates['y_coord']
    ds_toSHP['prediction'] = y_hay
    return ds_toSHP


## From here on NOT READY !!!
def clipRasterWithPoligon(rastPath, polygonPath,outputPath):
    '''
    Clip a raster (*.GTiff) with a single polygon feature 
    '''
    os.system("gdalwarp -datnodata -9999 -q -cutline" + polygonPath + " crop_to_cutline" + " -of GTiff" + rastPath + " " + outputPath)
   
   
def separateClippingPolygonss(inPath,field, outPath = "None"):
    '''
    Crete individial *.shp for each Clip in individual directories. 
    @input: 
       @field: Flield in the input *.shp to chose.
       @inPath: The path to the original *.shp.
    '''
    if outPath != "None":
        ensureDirectory(outPath)
        os.mkdir(os.path.join(outPath,"/clipingPolygons"))
        saveingPath = os.path.join(outPath,"/clipingPolygons") 
    else: 
        ensureDirectory(os.path.join(getLocalPath(),"/clipingPolygons"))
        saveingPath = os.path.join(outPath,"/clipingPolygons")

    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(inPath)
    if ds in None:
        print("Layer not found")
        return False
    lyr = ds.GetLayer()
    spatialRef = lyr.GetSpatialRef()
    for feautre in lyr:
        fieldValue = feautre.GetField(field)
        os.mkdir(os.path.join(saveingPath,str(fieldValue)))
        outName = str(fieldValue)+"Clip.shp"
        outds = driverSHP.CreateDataSorce("clipingPolygons/" + str(fieldValue) + "/" + outName)
        outLayer = outds.CreateLayer(outName, srs=spatialRef,geo_type = ogr.wkbPolygon)
        outDfn = outLayer.GetLayerDef()
        inGeom = feautre.GetGeometryRef()
        outFeature = ogr.Feature(outDfn)
        outFeature.SetGeometry(inGeom)
        outLayer.CreateFeature(outFeature)
    
    return True

def clipRaster(rasterPath,polygonPath,field, outputPath):
    ''' 
    '''
    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(polygonPath)
    if ds in None:
        print("Layer not found")
        return False
    lyr = ds.GetLayer()
    spatialRef = lyr.GetSpatialRef()
    for feautre in lyr:
        fieldValue = feautre.GetField(field)
        clipRasterWithPoligon(rasterPath, polygonPath, outputPath)
    return True
    
