import Globals
from tkinter import filedialog, messagebox
import os
import pandas
import openpyxl
import numpy
from scipy.sparse import data
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import time
import matplotlib
import joblib
import pickle

def loadDataset() -> None:
    datasetSelection = filedialog.askopenfile(title="Please select a dataset file", initialdir=os.getcwd(), filetypes=[('CSV Files', '*.csv'),('Excel Files', '*.xlsx')])
    
    if(datasetSelection == None or datasetSelection == ''): #Cancel, as nothing was selected.
        if(Globals.dataset is None):
            Globals.datasetPath = ""
            Globals.selectedFile = "Select File"
            return 'Cancel', Globals.selectedFile
        else:
            return 'Cancel', Globals.selectedFile
        
    elif(".xslx" in datasetSelection.name or ".xls" in datasetSelection.name):
        try:
            Globals.dataset = pandas.read_excel(datasetSelection.name, engine="openpyxl")
        except Exception as error:
            messagebox.showerror("Parse Dataset Error", error)
            Globals.dataset = None
    elif ".csv" in datasetSelection.name:
        try:
            Globals.dataset = pandas.read_csv(datasetSelection.name)
        except Exception as error:
            messagebox.showerror("Parse Dataset Error", error)
            Globals.dataset = None
            
    else:
        Globals.dataset = None
        messagebox.showerror("Dataset Type Error", "Invalid type, please try again.")
    
    #Dataset is too small
    if(Globals.dataset is not None and len(list(Globals.dataset)) < 2):
        messagebox.showerror("Dataset to Small Error", "Not enough data to train an AI model.\nPlease use a dataset with 2 or more columns!")
        Globals.dataset = None

    if(Globals.dataset is None):
        Globals.datasetPath = ""
        Globals.selectedFile = "Select File"
        return 'False', Globals.selectedFile
    Globals.selectedFile = str(datasetSelection.name)[datasetSelection.name.rfind('/') + 1:]
    Globals.datasetPath = str(datasetSelection.name)
    if(len(Globals.selectedFile) > 29):
        Globals.selectedFile = Globals.selectedFile[0:22] + "..." + Globals.selectedFile[-4:]
    
    Globals.resetArrays()
    
    Globals.originalColumnNames = list(Globals.dataset)
    Globals.originalRowNumber = Globals.dataset.shape[0]
    #print(Globals.originalColumnNames)
    #print(Globals.originalRowNumber)

    return 'True', Globals.selectedFile

"""
        Process Columns Section
"""
#region Process Columns Section
def viewColumns(hideTargetedColumns) -> list:
    if(not hideTargetedColumns):
        return list(Globals.dataset)
    
    returnList = list(Globals.dataset)
    for index in list(Globals.dataset):
        if(index in Globals.targetColumnsList):
            returnList.remove(index)
    return returnList

def processTargetedColumns(selectedList) -> bool:
    if(len(selectedList) == 0):
        messagebox.showerror("No Target Columns Error", "Please select at least 1 column as a Target for training.")
        return False
    if(len(list(Globals.dataset)) - len(selectedList) < 1):
        messagebox.showerror("Target Columns Error", "Not enough data to train an AI model.\nPlease leave at least 1 column of the original dataset.")
        return False
    
    Globals.targetColumnsList = list(selectedList)
    Globals.targetColumnsIndex = []
    Globals.rawInputsColumnsIndex = []
    Globals.ignoredColumnsIndex = []
    Globals.ignoredColumnsList.clear()
    for column in Globals.targetColumnsList: #The AI will predict target values
        try:
            Globals.targetColumnsIndex.append(list(Globals.dataset).index(column))
        except:
            continue
    for column in range(len(list(Globals.dataset))): #The AI will predict target values using these remaining input values
        if(column in Globals.targetColumnsIndex):
            continue
        Globals.rawInputsColumnsIndex.append(column)
    Globals.finalInputsColumnsIndex = Globals.rawInputsColumnsIndex.copy()
    
    #print("\nTargets:", Globals.targetColumnsIndex, Globals.targetColumnsList)
    #print("Inputs (Raw):", Globals.rawInputsColumnsIndex)
    #print("Inputs (Final):", Globals.finalInputsColumnsIndex)
    #print("Ignoring:", Globals.ignoredColumnsIndex, Globals.ignoredColumnsList)
    return True

def processIgnoredColumns(selectedList) -> bool:
    if(len(list(Globals.dataset)) - len(selectedList) < 1):
        messagebox.showerror("Remove Columns Error", "Not enough data to train an AI model.\nPlease leave at least 1 column of the original dataset.")
        return False

    Globals.ignoredColumnsList = list(selectedList)
    Globals.ignoredColumnsIndex = []
    Globals.finalInputsColumnsIndex = Globals.rawInputsColumnsIndex.copy()
    for column in Globals.ignoredColumnsList:
        try:
            index = list(Globals.dataset).index(column)
            Globals.ignoredColumnsIndex.append(index)
            Globals.finalInputsColumnsIndex.remove(index)
        except Exception as error:
            #print("Continuing", error)
            continue
    
    #print("\nTargets:", Globals.targetColumnsIndex, Globals.targetColumnsList)
    #print("Inputs (Raw):", Globals.rawInputsColumnsIndex)
    #print("Inputs (Final):", Globals.finalInputsColumnsIndex)
    #print("Ignoring:", Globals.ignoredColumnsIndex, Globals.ignoredColumnsList)
    return True
#endregion

"""
        Reformat Help Section
"""
#region Reformat Help Section
def renameColumnSelectorButtons(columnList = None) -> str:
    if(columnList == None or columnList == "" or columnList == []):
        return "None"
    
    numberOfColumns = len(columnList)
    rewrittenFirstCol = columnList[0]
    if(len(rewrittenFirstCol) > 17): #String is too long, so shorten it.
        rewrittenFirstCol = rewrittenFirstCol[0:14] + "..."

    if(numberOfColumns > 1):
        return rewrittenFirstCol + " + " + str(numberOfColumns - 1) + " more"
    return rewrittenFirstCol

def reformatFirstRow(firstRowIsData):
    renameColumns = []
    if(firstRowIsData): #RESAVE COLUMN NAMES TO BE GENERIC
        for col in range(len(list(Globals.dataset))):
            renameColumns.append("Col" + str(col))
            
        #Add another row because the original column names are supposed to be cells in the first row
        Globals.dataset.loc[-1] = Globals.originalColumnNames
        Globals.dataset.index = Globals.dataset.index + 1
        Globals.dataset.sort_index(inplace=True)
    else: #RESAVE COLUMN NAMES TO BE ORIGINAL (Labels)
        renameColumns = Globals.originalColumnNames
        
        if(Globals.dataset.shape[0] >= Globals.originalRowNumber + 1): #An extra row has been added, so remove it
            Globals.dataset = Globals.dataset.iloc[1:]

    Globals.targetColumnsList.clear()
    for col in Globals.targetColumnsIndex:
        Globals.targetColumnsList.append(renameColumns[col])
            
    Globals.ignoredColumnsList.clear()
    for col in Globals.ignoredColumnsIndex:
        Globals.ignoredColumnsList.append(renameColumns[col])
        
    Globals.dataset.columns = renameColumns
    
    #print("\nTargets:", Globals.targetColumnsIndex, Globals.targetColumnsList)
    #print("Ignoring:", Globals.ignoredColumnsIndex, Globals.ignoredColumnsList)
    #print(Globals.originalColumnNames)
    #print(Globals.originalRowNumber)
    return Globals.targetColumnsList, Globals.ignoredColumnsList
#endregion

"""
        Model Section
"""
#region Model Section
def trainModel(modelSettings, cleanSettings, doneFunction) -> None:
    if(Globals.dataset is None):
        return
    if(len(modelSettings) <= 0 or len(cleanSettings) <= 0):
        doneFunction(False, "\nError with Model or Cleaning Section.\n", '')
        return
    if(len(Globals.targetColumnsIndex) <= 0):
        return

    dataTable = Globals.dataset.copy()
    suggestions = ""
    results = ""
    
    Globals.modelUsedModelSettings = modelSettings
    Globals.modelUsedCleanSettings = cleanSettings
    
    if(cleanSettings[0]): #Reformat string columns to integer values
        reformattedColumnsCount = 0
        for col in Globals.originalColumnNames:
            if col in Globals.ignoredColumnsList: #Don't include ignored columns
                continue
            if(Globals.dataset[col].dtype == 'object'):
                reformattedColumnsCount += 1
                codes, uniques = pandas.factorize(dataTable[col])
                dataTable[col] = codes
        if(reformattedColumnsCount > 0):
            suggestions += "Change text columns into numbers: " + str(reformattedColumnsCount) + " columns.\n"
                
    if(dataTable.isnull().values.any()): #Is there a spot where a cell is null?
        suggestions += "Fix cells with missing values: " + str(dataTable.isnull().sum()) + " cells.\n"
        if(cleanSettings[1]): #Fill median
           for col in Globals.originalColumnNames:
               dataTable[col].fillna(dataTable[col].median(), inplace=True)
        else: #Drop row
           dataTable.dropna(axis=0, how='any')
            
    if(cleanSettings[2]): #Resaved Cleaned Dataset
        saveTable = dataTable.copy()
        if(len(Globals.ignoredColumnsList) > 0): #Drop ignored columns
            saveTable.drop(columns=Globals.ignoredColumnsList, inplace=True)
        try:
            saveTable.to_csv(str(Globals.datasetPath[0:len(Globals.datasetPath)-4]) + "Cleaned.csv", index=False)
        except:
            messagebox.showerror("Unable to save cleaned dataset", "Unable to saved the cleaned dataset.\nThis could be caused by leaving the cleaned dataset open inside a different program.")
        
    dataTable = dataTable.to_numpy()

    targets = dataTable[:,Globals.targetColumnsIndex]
    inputs = dataTable[:,Globals.finalInputsColumnsIndex]
    trainInputs, testInputs, trainTargets, testTargets = train_test_split(inputs, targets, test_size=modelSettings[6], random_state=5)
    if(modelSettings[6] > 90):
        suggestions += "Small training size, consider lowering test size.\n"
    elif(modelSettings[6] < 10):
        suggestions += "Small test size, consider raising test size.\n"

    try:
        if(modelSettings[0] == 'scikit-learn'):
            model = MLPRegressor(solver='sgd', #lbfgs
                                  hidden_layer_sizes=(modelSettings[1], modelSettings[2]), 
                                  learning_rate_init=modelSettings[3],
                                  alpha=modelSettings[4], 
                                  activation='tanh', 
                                  random_state=5, 
                                  max_iter=modelSettings[5], 
                                  early_stopping=True)
            startTime = time.time()
            model.fit(trainInputs, trainTargets)
            model.score(testInputs, testTargets)
            
            Globals.trainedModelGraph = model.loss_curve_
            
            results += "Model finished training in " + str(round(time.time() - startTime, 2)) + " seconds"
            results += "\nLoss: " + str(round(model.loss_, 6))
            results += "\nBest Validation Score: " + str(round(model.best_validation_score_, 6))
        #elif(modelSettings[0] == 'Tensorflow'):
            #model = 
            
        Globals.trainedModel = model
        
        perm_importance = permutation_importance(model, testInputs, testTargets, n_repeats=25, random_state=5)
        feature_importance = perm_importance.importances_mean
        importance_ids = feature_importance.argsort()
        results += "\nMost Important Column: " + Globals.originalColumnNames[Globals.finalInputsColumnsIndex[importance_ids[-1]]]
        if(feature_importance[importance_ids[0]] < 0):
            suggestions += "Negative Importance Column: " + Globals.originalColumnNames[Globals.finalInputsColumnsIndex[importance_ids[0]]]
            
            
        doneFunction(True, results, suggestions)
            
    except Exception as error:
        messagebox.showerror("Error from training", error)
        doneFunction(False, f"Error when training.\n{str(error)[0:50]}", '')

def predictWithModel(stringvar) -> str:
    inputString = stringvar.get()

    if(Globals.trainedModel is None):
        return "Model is not trained!"
    
    trainedColumnLength = len(Globals.finalInputsColumnsIndex)
    if(trainedColumnLength <= 0):
        return "Prediction: N/A"  

    userInput = inputString.replace(" ", "") #Remove any spaces
    userInput = userInput.split(",") #Split as array
    if(type(userInput) == str): #There was only one column, so save as array
        #print("It is a string, not array!")
        userInput = [userInput]
    
    if(len(userInput) <= 0):
        messagebox.showerror("No data", "Please enter data before trying to predict!")
        return "Prediction: N/A" 

    if(not (len(userInput) == trainedColumnLength)): #Array is not big enough?
        messagebox.showerror("Columns in Interact does not match dataset", "You do not have the same amount of columns as the used dataset!" +
                             "\nThe used dataset is " + str(trainedColumnLength) + " columns." +
                             "\nMake sure to have " + str(trainedColumnLength - 1) + " commas as separators.")
        return "Error: Incorrect number of columns"

    for cell in range(0, len(userInput)):
        try:
            userInput[cell] = float(userInput[cell])
        except:
            messagebox.showerror("Error in Interact", str(userInput[cell]) + " is not a valid number!")
            return "Error: Invalid number"
    
    prediction = str(Globals.trainedModel.predict(numpy.array(userInput).reshape(1, -1)))
    return "Predicted Result: " + prediction
#endregion

"""
        Saving Section
"""
#region Saving Section
def createFilePath():
    filePath = filedialog.asksaveasfilename(filetypes=[("Pickle Dumps", "*.pkl")])
    if(filePath == ''):
        return ''
    if(filePath[-4:] != ".pkl"):
        filePath += ".pkl"
    return filePath

def saveTrainedModel():
    try:
        filePath = createFilePath()
        if(filePath == ''):
            return
        joblib.dump(Globals.trainedModel, filePath)
        messagebox.showinfo("Model file created at " + filePath, f"To load this model into Python code, use: model = joblib.load('{filePath}')")
    
    except Exception as error:
        messagebox.showerror("Error when saving Model", str(error))
        
def saveInstance(resultsText, suggestionsText):
    saveList = []
    saveList.append("AIWorkbench Instance")
    saveList.append(Globals.dataset)
    saveList.append(Globals.originalColumnNames)
    saveList.append(Globals.originalRowNumber)

    saveList.append(Globals.targetColumnsList)
    saveList.append(Globals.ignoredColumnsList)
    
    saveList.append(Globals.targetColumnsIndex)
    saveList.append(Globals.rawInputsColumnsIndex)
    saveList.append(Globals.finalInputsColumnsIndex)
    saveList.append(Globals.ignoredColumnsIndex)
    
    saveList.append(Globals.trainedModel)
    saveList.append(Globals.trainedModelGraph)  

    saveList.append(Globals.modelUsedModelSettings)
    saveList.append(Globals.modelUsedCleanSettings)
    
    saveList.append(resultsText)
    saveList.append(suggestionsText)
    try:
        filePath = createFilePath()
        if(filePath == ''):
            return
        joblib.dump(saveList, filePath)
        
    except Exception as error:
        messagebox.showerror("Error when saving Instance", str(error))
        
def loadInstance() -> tuple:
    filePath = filedialog.askopenfile(title="Please select a dataset file", initialdir=os.getcwd(), filetypes=[("Pickle Dumps", "*.pkl")])
    if(filePath == '' or filePath == None): #Cancel, as nothing was selected.
        return False, [], [], '', ''
    try:
        saveList = joblib.load(filePath.name)
        if(saveList[0] != "AIWorkbench Instance" or len(saveList) != 16):
            messagebox.showerror("Invalid Instance", "The file you chose was not a recognized instance.\nPlease select a different file.")
            return False, [], [], '', ''
        Globals.dataset = saveList[1]
        Globals.originalColumnNames = saveList[2]
        Globals.originalRowNumber = saveList[3]
        
        Globals.targetColumnsList = saveList[4]
        Globals.ignoredColumnsList = saveList[5]
        
        Globals.targetColumnsIndex = saveList[6]
        Globals.rawInputsColumnsIndex = saveList[7]
        Globals.finalInputsColumnsIndex = saveList[8]
        Globals.ignoredColumnsIndex = saveList[9]
        
        Globals.trainedModel = saveList[10]

        Globals.trainedModelGraph = saveList[11]
        
        Globals.modelUsedModelSettings = saveList[12]
        Globals.modelUsedCleanSettings = saveList[13]
        
        Globals.selectedFile = str(filePath.name)[filePath.name.rfind('/') + 1:]
        Globals.datasetPath = str(filePath.name)
        if(len(Globals.selectedFile) > 29):
            Globals.selectedFile = Globals.selectedFile[0:22] + "..." + Globals.selectedFile[-4:]
        
        return True, saveList[12], saveList[13], saveList[14], saveList[15]
        
    except Exception as error:
        if("'MLPRegressor' object is not subscriptable" in str(error)):
            messagebox.showerror("Invalid Instance", "The file you selected was a model, not an instance.\nPlease select a instance.")
        elif('object is not subscriptable' in str(error)):
            messagebox.showerror("Invalid Instance", "The file you chose was not a recognized instance.\nPlease select a instance.")
        else:
            messagebox.showerror("Error when loading Instance", str(error))
        
    return False, [], [], '', ''
#endregion