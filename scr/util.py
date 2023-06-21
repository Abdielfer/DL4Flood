import os
import glob
import pathlib
import shutil
import joblib
import time
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader,random_split

import rasterio as rio
from rasterio.plot import show_hist
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

def makeNameByTime()->str:
    name = time.strftime("%y%m%d%H%M")
    return name

## Model manipulation
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

####  Sampling and Dataset manipulation
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

def randomSamplingFromList(listeToSmpl, numberOfSmpl)->int:
    '''
    @Retiurn <numberOfSmpl> 0random index from <listeToSmpl>
    '''
    return random.sample(listeToSmpl, numberOfSmpl)

def splitPerRegion(csvPath, trnFract:float = .95)->list:
    fullList = createListFromCSV(csvPath) 
    imagList = createListFromCSVColumn(csvPath,0,delim=';')
    lenimagList = len(imagList)
    print("Total samples >>", len(imagList))
    parent,_,_ = get_parenPath_name_ext(imagList[0])
    start = parent
    print(f"First assignement: {start != parent} \n {start}\n{parent}")
    print(f"{start != parent} >> {start}>{parent}")
    regionCounter = 0
    lastParent,_,_ = get_parenPath_name_ext(imagList[-1])
    allPath = []
    trn = []
    tst = []
    i = 0
    while start == parent:
        parent,_,_ = get_parenPath_name_ext(imagList[i])
        if start != parent:
            regionCounter+=1
            trnToAdd, tstToAdd = distrubutePathInTrnTst(allPath,trnFract=trnFract)
            trn.extend(trnToAdd)
            tst.extend(tstToAdd)
            # print(f"Training set evolution : {len(trn)} vs Teste set {len(tst)} >> Total {len(trn)+len(tst)}")
            # print("_____________________________")
            start = parent
            allPath = []
            if start == lastParent:
                regionCounter+=1
                allPath = imagList[i:-1]
                trnToAdd, tstToAdd = distrubutePathInTrnTst(allPath,trnFract=trnFract)
                trn.extend(trnToAdd)
                tst.extend(tstToAdd)
                # print(f"Training set evolution : {len(trn)} vs Teste set {len(tst)} >> Total {len(trn)+len(tst)}")
                print(f"We finished sampling {regionCounter} regions")
                break
            continue
        allPath.append(fullList[i])
        i +=1
        if i == lenimagList: break
    return trn,tst

def distrubutePathInTrnTst(allSampesPath,trnFract:float = 0.9)->list:
    trnPErcent = int(len(allSampesPath)*trnFract)
    srandomSamples = randomSamplingFromList(allSampesPath,trnPErcent) 
    trn, tst = [],[]
    for item in srandomSamples:
        trn.append(allSampesPath.pop(allSampesPath.index(item)))
    tst = allSampesPath
    return trn,tst
  
def importDataSet(dataSetName, targetCol: str):
    '''
    @input: DataSetName => The dataset path. DataSet must be in *.csv format. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(dataSetName, index_col = None)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    return x, y

def splitDataset(dataset:Dataset, proportions = [.9,.1] ,seed:int = 42, )-> Dataset:
    '''
    ref: https://pytorch.org/docs/stable/data.html# 
    '''
    len = dataset.__len__()
    lengths = [int(p *len) for p in proportions]
    lengths[-1] = len - sum(lengths[:-1])
    generator = torch.Generator().manual_seed(seed)
    train_CustomDS, val_CustomDS = random_split(dataset,lengths,generator=generator)
    return train_CustomDS, val_CustomDS

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
    # print(content)    
    return content

def getLocalPath():
    return os.getcwd()

def makePath(str1,str2):
    return os.path.join(str1,str2)

def ensureDirectory(pathToCheck):
    if not os.path.isdir(pathToCheck): 
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
        return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath must contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

def makeFileCopy(inputFilePath, outputFilePath):
    try:
        shutil.copy(inputFilePath, outputFilePath)
        return outputFilePath
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    except PermissionError:
        print("Permission denied.")
    except:
        print("Error occurred while copying file.")

def removeFile(filePath):
    try:
        os.remove(filePath)
        return True
    except OSError as error:
        print(error)
        print("File path can not be removed")
        return False

def createTransitFolder(parent_dir_path, folderName:str = 'TransitDir'):
    path = os.path.join(parent_dir_path, folderName)
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

def listFreeFilesInDirByExt(cwd, ext = '.tif'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    file_list = []
    for (root, dirs, file) in os.walk(cwd):
        for f in file:
            _,_,extent = get_parenPath_name_ext(f)
            if extent == ext:
                file_list.append(f)
    return file_list

def listFreeFilesInDirByExt_fullPath(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            _,extent = splitFilenameAndExtention(f)
            if ext == extent:
                file_list.append(os.path.join(root,f))
    return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList: list = []
    for (root, _, _) in os.walk(cwd):
         fullList.extend(listFreeFilesInDirByExt(root, ext)) 
    return fullList

def listALLFilesInDirByExt_fullPath(cwd, ext = '.csv'):
    '''
    @ext: NOTE <ext> must contain the "." ex: '.csv'; '.tif'; etc...
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirByExt_fullPath(root, ext)
        # print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
    return fullList

def createListFromCSV(csv_file_location, delim:str =','):  
    '''
    @return: list from a <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    '''       
    df = pd.read_csv(csv_file_location, index_col= None, delimiter = delim)
    out = []
    for i in range(0,df.shape[0]):
        out.append(df.iloc[i,:].tolist()[0])    
    return out

def createListFromCSVColumn(csv_file_location, col_idx:int, delim:str =','):  
    '''
    @return: list from <col_id> in <csv_file_location>.
    Argument:
    @col_index: 
    @csv_file_location: full path file location and name.
    @col_idx : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.read_csv(csv_file_location, index_col= None, delimiter = delim)
    fin = df.shape[0]
    for i in range(0,fin):
        x.append(df.iloc[i,col_idx])
    return x

def createListFromExelColumn(excell_file_location,Sheet_id:str, col_idx:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_idx]:
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

def get_parenPath_name_ext(filePath):
    '''
    Ex: user/folther/file.txt
    parentPath = pathlib.PurePath('/src/goo/scripts/main.py').parent 
    parentPath => '/src/goo/scripts/'
    parentPath: can be instantiated.
         ex: parentPath[0] => '/src/goo/scripts/'; parentPath[1] => '/src/goo/', etc...
    '''
    parentPath = pathlib.PurePath(filePath).parent
    name, ext = splitFilenameAndExtention(filePath)
    return parentPath, name, ext
  
def addSubstringToName(path, subStr: str, destinyPath = None):
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,name,ext= get_parenPath_name_ext(path)
    if destinyPath != None: 
        newPath = os.path.join(destinyPath,(name+subStr+ext))
    else: 
        newPath = os.path.join(parentPath,(name+subStr+ ext))
    return newPath

def createCSVFromList(pathToSave: os.path, listData:list):
    '''
    Ths function create a *.csv file with one line per <lstData> element. 
    @pathToSave: path of *.csv file to be writed with name and extention.
    @listData: list to be writed. 
    '''
    parentPath,name,_ = get_parenPath_name_ext(pathToSave)
    textPath = makePath(parentPath,(name+'.txt'))
    with open(textPath, 'w') as output:
        for line in listData:
            output.write(str(line) + '\n')
    read_file = pd.read_csv (textPath)
    print(f'Creating CSV at {pathToSave}')
    read_file.to_csv (pathToSave, index=None)
    removeFile(textPath)
    return True

def makeTifGpkgPairsList(filesPath, delim:str = ',', mode: str = 'trn'):
    print(filesPath)
    tifList = listFreeFilesInDirByExt_fullPath(filesPath, ext='.tif')
    gpkgList = listFreeFilesInDirByExt_fullPath(filesPath,ext='.gpkg')
    pairImgMask = []
    for tif in tifList:
        tifName,_ = splitFilenameAndExtention(tif)
        for gpkg in gpkgList:
            if tifName in gpkg:
                tifPath = makePath(filesPath,tif)
                gpkgPath = makePath(filesPath, gpkg)
                str4List = tifPath + delim + gpkgPath + delim + mode
                pairImgMask.append(str4List)
    return pairImgMask

def noMatch_TifMask_List(scvPath,tifDir,col_idx:int=0,delim:str =',',
                         relocate: bool = False, relocatePath:os.path = None)->list:
    '''
    Finde the *.tif into the <tifDir> without match into the list of tif-mask pairs.
    @csvPath: *csv containing the list of tif-mask pairs. 
    @return: list of no matching file's names. NOTE: If relocate = True, retur a list of full paths. 
    @relocate: Bool: Defines if the no matching files are immediately relocated to a TransitFolder.
    '''
    csvList = createListFromCSVColumn(scvPath,col_idx,delim = delim)
    print(f"CsV list len {len(csvList)}")
    if relocate == False:
        print("Relocate False")
        tifList = listALLFilesInDirByExt(tifDir, ext='.tif')
        noMatchList = [tif for tif in tifList if not any([tif in item for item in csvList])] 
        print(f"Available *tif :  {len(tifList)}")
        print(f"NO matchig *.tif : {len(noMatchList)}")
        return noMatchList
    elif relocate == True:
        print("Relocate True")
        tifList = listALLFilesInDirByExt_fullPath(tifDir, ext='.tif')
        print(f"tifList list len {len(tifList)}")
        noMatchList = []
        if relocatePath == None:
            relocatePath = input("Enter the path to relocate files")
        transitPath = createTransitFolder(relocatePath)
        for tif in tifList:
            _,tifName,_ = get_parenPath_name_ext(tif)
            if not any([tifName in item for item in csvList]):
                noMatchList.append(tif)
                relocateFile(tif,transitPath)
        print(f"Available *tif :  {len(tifList)}")
        print(f"NO matchig *.tif : {len(noMatchList)}")
        return noMatchList
    
def makeMatching_TifMask_List(scvPath,tifDir,delim:str =',',)->list:
    '''
    Finde the *.tif into the <tifDir> without match into the list of tif-mask pairs.
    @csvPath: *csv containing the list of tif-mask pairs. 
    @return: list of no matching file's names. NOTE: If relocate = True, retur a list of full paths. 
    @relocate: Bool: Defines if the no matching files are immediately relocated to a TransitFolder.
    '''
    matchingList =[]
    tifList = listALLFilesInDirByExt(tifDir, ext='.tif')
    matchingList = [tif for tif in tifList if any([tif in item for item in csvList])] 
    print(f"Matching list len: -> {len(matchingList)}")   
    csvList = createCSVFromList(scvPath,matchingList, delim = delim)
    return matchingList
   
## Device
def checkDevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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
def plotImageAndMask(img, mask,imgName:str='Image', mskName: str= 'Mask'):
    # colList = ['Image','Mask']
    image = img.detach().numpy() if torch.is_tensor(img) else img.numpy().squeeze()
    mask_squeezed = mask.detach().numpy() if torch.is_tensor(mask) else mask.numpy().squeeze()
    fig, axs = plt.subplots(1,2, figsize=(10,5), sharey=True)
    axs[0].imshow(image, cmap='Greys_r')
    axs[0].set(xlabel= imgName)
    axs[1].imshow(mask_squeezed, cmap='Greys_r')
    axs[1].set(xlabel= mskName)
    plt.rcParams['font.size'] = '15'
    fig.tight_layout()
     
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

def imageToTensor(img,DTYPE:str = 'float32'):
    imagTensor = img.astype(DTYPE)
    # imagTensor = np.transpose(imagTensor, (2, 0, 1)).astype(DTYPE)
    imagTensor = torch.tensor(imagTensor)
    return imagTensor

def reshape_as_image(arr):
    '''
    From GDL
    Parameters
    ----------
    arr : arr as image in raster order (bands, rows, columns)
    return: array-like in the image form of (rows, columns, bands)
    '''       
    return np.ma.transpose(arr, [1, 2, 0]).astype('float32')

def reshape_as_raster(arr):
    '''  
    From GDL
        swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    return: arr as image in raster order (bands, rows, columns)
    '''
    return np.transpose(arr, [2, 0, 1])

def makePredictionRaster(rasterPath:os.path, model, saveRaster:bool=False):
    '''
    Crete a raster prediction with the same metadata as the inputRaster.
    params:
     @input: rasterPath
     @autput: 
    '''
    # Name and savePath creation
    parentPath,name,_ = get_parenPath_name_ext(rasterPath)
    savePath = os.path.join(parentPath, (name+'_predicted.tif'))
    # read raster
    data, profile = readRaster(rasterPath)
    # print(profile)
    rasterData = imageToTensor(data)[None,:] # if model demand extra dimention
    print('rasterData.shape into makePredictionRaster', rasterData.shape)
    # model
    model.eval()
    y_hat = model(rasterData).detach().numpy()
    rasterData = y_hat[0][0]
    model.train()
    # Save raster
    if saveRaster:
        createRaster(savePath,rasterData, profile)    
    return y_hat[0]  #remouve the extra dimention added to pass through the model. 

def readRaster(rasterPath:os.path):
    '''
    Read a raster qith Rasterio.
    return:
     Raster data as np.array
     Raster.profile: dictionary with all rater information
    '''
    inRaster = rio.open(rasterPath, mode="r")
    profile = inRaster.profile
    rasterData = inRaster.read()
    # print(f"raster data shape in ReadRaster : {rasterData.shape}")
    return rasterData, profile

def createRaster(savePath:os.path, data:np.array, profile, noData:int = None):
    '''
    parameter: 
    @savePath: Most contain the file name ex.: *name.tif.
    @data: np.array with shape (bands,H,W)
    '''
    B,H,W = data.shape[-3],data.shape[-2],data.shape[-1] 
    # print(f"C : {B}, H : {H} , W : {W} ")
    profile.update(dtype = rio.uint16, nodata = noData, blockysize = profile['blockysize'])
    with rio.open(
        savePath,
        mode="w",
        #out_shape=(B, H ,W),
        **profile
        ) as new_dataset:
            # print(f"New Dataset.Profile: ->> {new_dataset.profile}")
            new_dataset.write(data)
            print("Created new raster>>>")
    return savePath
   
def plotHistogram(raster, bins: int=50, bandNumber: int = 1):
    show_hist(source=raster, bins=bins, title= f"Histogram of {bandNumber} bands", 
          histtype='stepfilled', alpha=0.5)
    return True

#### Compute globals Mean and STD from a set of raster images###
class standardizer():
    def __init__(self) -> None:
        '''
        Class to manage the standardization of a set of rasters. 
         - Compute the global values of min, max, mean and STD for a set of raster.
         - Save the global values as *csv file.
         - Standardize a raster set or a single raster, using the globals min, max, mean and std
        '''
        self.globMin = np.inf
        self.globMax = -np.inf
        self.globMean = 0
        self.globSTD = 0
       
    def computeGlobalValues(self, rasterListPath, extraNoData = None):
        '''
        Compute the global Standard Deviation from a list of raters.
        @rasterList: a list of raster path.
        '''
        rasterList = createListFromCSVColumn(rasterListPath,0,';')
        globalCont = 0  # The total number of pixels in <rasterList>, different from NoData value.
        cummulativeMean = 0
        # Compute globalMin, globalMax, globalMean
        for ras in rasterList:
            localMin, localMax,rasMean,rasCont = computeRaterStats(ras) #rasMin, rasMax, rasMean, rasNoNaNCont
            self.updateGlobalMinMax(localMin, localMax)
            globalCont += rasCont
            cummulativeMean += rasMean
        self.globMean = cummulativeMean/len(rasterList)  # From the math principle: the mean of subsets means is also the global mean. 
        print(f"Globals : min:{self.globMin}>> max:{self.globMax} >> MEan : {self.globMean} >> globalCount {globalCont}" )
        #Compute globa quadratic error
        globSumQuadraticError = 0 
        for raster in rasterList:
            rasData = replaceRastNoDataWithNan(raster,extraNoDataVal= extraNoData)
            globSumQuadraticError += computeSumQuadraticError(rasData,self.globMean)
        
        self.globSTD = math.sqrt(globSumQuadraticError/globalCont )
        print(f"Final values: GlobSumSQError {globSumQuadraticError}, GlobSTD : {self.globSTD}")

    def updateGlobalMinMax(self, localMin,localMax):
        if localMin < self.globMin: self.globMin = localMin
        if localMax > self.globMax: self.globMax = localMax

    def setGlobals(self,min = None, max = None, mean = None, std = None):
        if min is not None: self.globMin = min
        if max is not None: self.globMax = max
        if mean is not None: self.globMean = mean
        if std is not None: self.globSTD = std

    def setExtraNoData(self, value):
        self.extraNoData = value

    def getGlobals(self):
        return self.globMin, self.globMax, self.globMean, self.globSTD   

    def saveGlobals(self, pathToSaveCSV: os.path):
        globals = {'globalMin': self.globMin, 'globalMax': self.globMax, 'globalMean': self.globMean, 'globalSTD': self.globSTD}
        with open(pathToSaveCSV, 'w') as f:
            for key in globals.keys():
                f.write("%s,%s\n"%(key,globals[key]))

    def standardizeRasterData(self, rasterData:np.array )->np.array:
        output = (rasterData - self.globMean)/self.globSTD 
        return  output

def replaceRastNoDataWithNan(rasterPath:os.path,extraNoDataVal: float = None)-> np.array:
    rasterData,profil = readRaster(rasterPath)
    NOData = profil['nodata']
    rasterDataNan = np.where(((rasterData == NOData)|(rasterData == extraNoDataVal)), np.nan, rasterData) 
    return rasterDataNan

def computeSumQuadraticError(arr:np.array, mean):
    '''
    Compute elementwise quadratic errors of a np.array, and its sum excluding np.nan values
    @mean: This mean could be the mean of <<arr>> or an external mean. 
    '''
    quadError = (arr - mean)**2
    return np.nansum(quadError)

def computeRaterStats(rasterPath:os.path):
    '''
    Read a reaste and return: 
    @Return
    @rasMin: Raster min.
    @rasMax: Raster max.
    @rasMean: Rater mean.
    @rasNoNaNSum: Raster sum of NOT NoData pixels
    @rasNoNaNCont: Raster count of all NOT NoData pixels
    '''
    rasDataNan = replaceRastNoDataWithNan(rasterPath)
    rasMin = np.min(rasDataNan)
    rasMax = np.max(rasDataNan)
    rasMean = np.mean(rasDataNan)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    return rasMin, rasMax, rasMean, rasNoNaNCont

def computeMatrixStats(input):
    '''
    @input: must be np.array or torch.tensor
    '''
    matrix = input.detach().cpu().numpy() if torch.is_tensor(input) else input.detach().cpu()
    return np.min(matrix), np.max(matrix), np.mean(matrix) 

## From hereon NOT READY !!!
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
    
