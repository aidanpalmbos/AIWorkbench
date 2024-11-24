import Functions
import Globals
import tkinter as tk
import os
from tkinter import IntVar, StringVar, Toplevel, ttk, messagebox
from idlelib.tooltip import Hovertip
import tkinter.font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from threading import Thread
import time

def getFont(_size=20, _weight='normal') -> tkinter.font:
    return tkinter.font.Font(family='Calibri', size=_size, weight=_weight)

def frameSetState(frame, enable):
    """Set the state of a frame and it's children, recursively."""
    for child in frame.winfo_children():
        if(isinstance(child, tk.Frame)): #If a frame is inside the current frame, become recursive.
            frameSetState(child, enable)
        else:
            if(isinstance(child, ttk.Progressbar)):
                child.config(mode='determinate')
                child.step(0)
            else:
                child.config(state='normal' if enable else 'disable')


root = tk.Tk()
childWindow = None

root.geometry("1050x650")
root.minsize(1050, 650)
root.title("AI Workbench")
iconPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "appIcon.ico"))
root.iconbitmap(iconPath)

columnInput = tk.Frame(root)
columnTrain = tk.Frame(root, highlightbackground='black', highlightthickness=1)
columnOutput = tk.Frame(root)

"""
        Input Column
"""
#region Input Column
labelInputColumn = tk.Label(columnInput, text='Input', height=1, font=getFont(_weight='bold'))
Hovertip(labelInputColumn, 'Column for determining the model and the data.', 350)
labelInputColumn.pack(fill='x')

#AI Frame
frameInputAI = tk.Frame(columnInput)
labelAI = tk.Label(frameInputAI, text='AI Model', font=getFont(15))
Hovertip(labelAI, 'Section for entering model settings.', 350)
labelAI.pack(pady=(10, 0))

subframeSelectPlatform = tk.Frame(frameInputAI)
labelSelectPlatform = tk.Label(subframeSelectPlatform, text='Train Regression on platform:', font=getFont(11))
optionsModel = ["scikit-learn"]
optionsModelWidth = len(max(optionsModel, key=len))
stringvarPlatform = StringVar()
stringvarPlatform.set(optionsModel[0])
selectPlatform = tk.OptionMenu(subframeSelectPlatform, stringvarPlatform, *optionsModel)
selectPlatform.config(width=optionsModelWidth, font=getFont(11))
labelSelectPlatform.pack(side=tk.LEFT, fill='x')
selectPlatform.pack(side=tk.LEFT, fill='x')
subframeSelectPlatform.pack(pady=5)

subframeHiddenLayer = tk.Frame(frameInputAI)
labelNumberOfLayers = tk.Label(subframeHiddenLayer, text='Hidden Layers:', font=getFont(11))
Hovertip(labelNumberOfLayers, 'Number of Hidden Layers of Neurons for training.\nGenerally the same number as the number of columns of your data.\nCannot be a fraction, 0, or negative.', 350)
stringvarNumberOfLayers = StringVar(value=11)
entryNumberOfLayers = tk.Entry(subframeHiddenLayer, textvariable=stringvarNumberOfLayers, width=4)
labelNumberOfNeurons = tk.Label(subframeHiddenLayer, text=', Neurons:', font=getFont(11))
Hovertip(labelNumberOfNeurons, 'Number of Neurons each layer has.\nToo few/Too many can Underfit/Overfit the data, respectively.\nCannot be a fraction, 0, or negative.', 350)
stringvarNumberOfNeurons = StringVar(value=80)
entryNumberOfNeurons = tk.Entry(subframeHiddenLayer, textvariable=stringvarNumberOfNeurons, width=4)
labelNumberOfLayers.pack(side=tk.LEFT, fill='x')
entryNumberOfLayers.pack(side=tk.LEFT, fill='x')
labelNumberOfNeurons.pack(side=tk.LEFT, fill='x')
entryNumberOfNeurons.pack(side=tk.LEFT, fill='x')
subframeHiddenLayer.pack(pady=5)

subframeRates = tk.Frame(frameInputAI)
labelInitialLearning = tk.Label(subframeRates, text='Initial Learning Rate:', font=getFont(11))
Hovertip(labelInitialLearning, 'Initial Learning rate of the neural network.\nDetermines how fast the network "learns" data, can increase loss.\nCannot be 0 or negative.', 350)
stringvarInitialLearning = StringVar(value=0.001)
entryInitialLearning = tk.Entry(subframeRates, textvariable=stringvarInitialLearning, width=6)
labelAlpha = tk.Label(subframeRates, text=', Alpha:', font=getFont(11))
Hovertip(labelAlpha, 'Value that reduces variance of predictions while training.\nIncreasing regulates it, while decreasing causes more variance.\nCannot be negative.', 350)
stringvarAlpha = StringVar(value=1e-10)
entryAlpha = tk.Entry(subframeRates, textvariable=stringvarAlpha, width=6)
labelInitialLearning.pack(side=tk.LEFT, fill='x')
entryInitialLearning.pack(side=tk.LEFT, fill='x')
labelAlpha.pack(side=tk.LEFT, fill='x')
entryAlpha.pack(side=tk.LEFT, fill='x')
subframeRates.pack(pady=5)

subframeIterationSplit = tk.Frame(frameInputAI)
labelIterations = tk.Label(subframeIterationSplit, text='Iterations:', font=getFont(11))
Hovertip(labelIterations, 'Maximum Iterations a model will try to train for.\nThe model will stop early if it reaches a constant loss.\nCannot be 0 or negative.', 350)
stringvarIterations = StringVar(value=10000)
entryIterations = tk.Entry(subframeIterationSplit, textvariable=stringvarIterations, width=6)
labelSplit = tk.Label(subframeIterationSplit, text=', % to use for testing:', font=getFont(11))
Hovertip(labelSplit, 'Split the dataset into training and testing for better validation.\nHigher values leaves less data for inputs, causing underfitting.\nLower values leaves less data for testing, causing overfitting.\nCannot be 0 or negative.', 350)
intvarSplit = IntVar(value=66)
entrySplit = tk.Spinbox(subframeIterationSplit, from_=1, to=100, textvariable=intvarSplit, width=3)
labelIterations.pack(side=tk.LEFT, fill='x')
entryIterations.pack(side=tk.LEFT, fill='x')
labelSplit.pack(side=tk.LEFT, fill='x')
entrySplit.pack(side=tk.LEFT, fill='x')
subframeIterationSplit.pack(pady=5)

#Data Frame
frameInputData = tk.Frame(columnInput)
labelData = tk.Label(frameInputData, text='Data', font=getFont(15))
Hovertip(labelData, 'Section for selecting a dataset and what the model needs to predict.', 350)
labelData.pack(pady=(10, 0))

subframeLoadData = tk.Frame(frameInputData)
labelLoadData = tk.Label(subframeLoadData, text='Load Dataset:', font=getFont(11))
Hovertip(labelLoadData, 'Select dataset to be used for training. Can be .csv, or .xlsx', 350)
buttonLoadData = tk.Button(subframeLoadData, text='Select File', font=getFont(11))
labelLoadData.pack(side=tk.LEFT, fill='x')
buttonLoadData.pack(side=tk.LEFT, fill='x')
subframeLoadData.pack(pady=5)
labelDataNote = tk.Label(frameInputData, text='Types supported: .xlsx, .csv', font=getFont(10))
labelDataNote.pack()

subframeTargetColumns = tk.Frame(frameInputData)
labelTargetColumns = tk.Label(subframeTargetColumns, text='Select Column(s) as Targets:', font=getFont(11))
Hovertip(labelTargetColumns, 'Select Columns of the dataset that the model will try to predict with the remaining columns.\nSelect multiple to have the model predict multiple columns.\nYou must select at least one.', 350)
buttonSelectTargetColumns = tk.Button(subframeTargetColumns, text='None', font=getFont(11))
labelTargetColumns.pack(side=tk.LEFT, fill='x')
buttonSelectTargetColumns.pack(side=tk.LEFT, fill='x')
subframeTargetColumns.pack(pady=5)

#Clean Frame
frameInputClean = tk.Frame(columnInput)
labelClean = tk.Label(frameInputClean, text='Clean Data', font=getFont(15))
Hovertip(labelClean, 'Section for modifying the dataset for proper training.', 350)
labelClean.pack(pady=(10, 0))

subframeFirstRow = tk.Frame(frameInputClean)
labelFirstRow = tk.Label(subframeFirstRow, text='The First Row contains:',  font=getFont(11))
Hovertip(labelFirstRow, 'If there are no labels to the columns in the dataset, choose data.\nOtherwise, choose labels.', 350)
boolCheckFirstRow = False
intvarFirstRow = IntVar(value=False)
radioFirstRowData = tk.Radiobutton(subframeFirstRow, text='Labels', font=getFont(11), variable=intvarFirstRow, value=False)
radioFirstRowLabel = tk.Radiobutton(subframeFirstRow, text='Data', font=getFont(11), variable=intvarFirstRow, value=True)
labelFirstRow.pack(side=tk.LEFT, fill='x')
radioFirstRowData.pack(side=tk.LEFT, fill='x')
radioFirstRowLabel.pack(side=tk.LEFT, fill='x')
subframeFirstRow.pack(pady=5)

subframeTextToNumbers = tk.Frame(frameInputClean)
labelTextToNumbers = tk.Label(subframeTextToNumbers, text='Change text values to numbers?', font=getFont(11))
Hovertip(labelTextToNumbers, 'When on, replace columns of text to numbers.\nA number is given to every value. For example, a text column containing either a or b will show as 0 or 1.', 350)
intvarTextToNumbers = IntVar(value=True)
checkboxTextToNumbers = tk.Checkbutton(subframeTextToNumbers, font=getFont(11), variable=intvarTextToNumbers)
labelTextToNumbers.pack(side=tk.LEFT, fill='x')
checkboxTextToNumbers.pack(side=tk.LEFT, fill='x')
subframeTextToNumbers.pack(pady=5)

subframeMissingData = tk.Frame(frameInputClean)
labelMissingData = tk.Label(subframeMissingData, text='Handle Missing Data:', font=getFont(11))
Hovertip(labelMissingData, 'If a cell is missing data, either remove the entire row or fill it with the median of that column.', 350)
intvarMissingData = IntVar(value=True)
radioMissingDataRemove = tk.Radiobutton(subframeMissingData, text='Remove row', font=getFont(11), variable=intvarMissingData, value=False)
radioMissingDataAverage = tk.Radiobutton(subframeMissingData, text='Fill median', font=getFont(11), variable=intvarMissingData, value=True)
labelMissingData.pack(side=tk.LEFT, fill='x')
radioMissingDataRemove.pack(side=tk.LEFT, fill='x')
radioMissingDataAverage.pack(side=tk.LEFT, fill='x')
subframeMissingData.pack(pady=5)

subframeIgnoreColumns = tk.Frame(frameInputClean)
labelIgnoreColumns = tk.Label(subframeIgnoreColumns, text='Select Column(s) to Ignore:', font=getFont(11))
Hovertip(labelIgnoreColumns, 'Allows you to select any columns that are not helpful or even harmful to leave out during training.', 350)
buttonSelectIgnoreColumns = tk.Button(subframeIgnoreColumns, text='None', font=getFont(11))
labelIgnoreColumns.pack(side=tk.LEFT, fill='x')
buttonSelectIgnoreColumns.pack(side=tk.LEFT, fill='x')
subframeIgnoreColumns.pack(pady=5)

subframeResave = tk.Frame(frameInputClean)
labelResave = tk.Label(subframeResave, text='Save copy of Cleaned Dataset?', font=getFont(11))
Hovertip(labelResave, 'When on, a copy of the cleaned dataset will be saved as [dataset]Cleaned.csv', 350)
intvarResave = IntVar(value=True)
checkboxResave = tk.Checkbutton(subframeResave, font=getFont(11), variable=intvarResave)
labelResave.pack(side=tk.LEFT, fill='x')
checkboxResave.pack(side=tk.LEFT, fill='x')
subframeResave.pack(pady=5)

#Load Model Frame
frameLoadInstance = tk.Frame(columnInput, highlightbackground='black', highlightthickness=1)
labelLoadInstance = tk.Label(frameLoadInstance, text='Or Load Previous Instance:', font=getFont(11, 'bold'))
Hovertip(labelLoadInstance, 'Load an instance file, created by the "Save Instance" button in the Output column.', 350)
buttonLoadInstance = tk.Button(frameLoadInstance, text='Select Instance', font=getFont(11))
labelLoadInstance.pack(side=tk.LEFT, fill='x')
buttonLoadInstance.pack(side=tk.LEFT, fill='x')

#PACK UP ALL FRAMES IN INPUT COLUMN
frameInputAI.pack(fill='y', expand=True)
frameInputData.pack(fill='y', expand=True)
frameInputClean.pack(fill='y', expand=True)
frameLoadInstance.pack(fill='y', expand=True)
#endregion

"""
        Train Column
"""
#region Train Column
#Begin Training Frame
frameBeginTraining = tk.Frame(columnTrain)
labelTrainColumn = tk.Label(frameBeginTraining, text='Train', height=1, font=getFont(_weight='bold'))
Hovertip(labelTrainColumn, 'Column for training and seeing the loss of the model.', 350)
labelTrainColumn.pack(fill='x')

buttonBeginTraining = tk.Button(frameBeginTraining, text='Begin Training', font=getFont(15, 'bold'))
buttonBeginTraining.pack(fill='x', pady=5, padx=20)

labelTrainingError = tk.Label(frameBeginTraining, text='No dataset selected.\nPlease selected a dataset.\nHover over text to get useful hints.', font=getFont(13))
labelTrainingError.pack(fill='both', expand=True)

progressBar = ttk.Progressbar(frameBeginTraining, mode='determinate')
progressBar.pack(fill='both', padx=20)

#Train Graph Frame
frameTrainGraph = tk.Frame(columnTrain)
figureWidget = Figure(figsize=(5,4), dpi=100)
plotWidget = figureWidget.add_subplot(111)
canvasWidget = FigureCanvasTkAgg(figureWidget, master=frameTrainGraph)
canvasWidget.draw()
canvasWidget.get_tk_widget().pack(fill='x')
toolbarWidget = NavigationToolbar2Tk(canvasWidget, frameTrainGraph)
toolbarWidget.update()
canvasWidget.get_tk_widget().pack(fill='x')

#PACK UP ALL FRAMES IN TRAIN COLUMN
frameBeginTraining.pack(fill='both', expand=True)
frameTrainGraph.pack(fill='x', pady=(4, 0))
#endregion

"""
        Output Column
"""
#region Output Column
labelOutputColumn = tk.Label(columnOutput, text='Output', height=1, font=getFont(_weight='bold'))
Hovertip(labelOutputColumn, 'Column for saving, interacting, and seeing the effectiveness of a trained model.', 350)
labelOutputColumn.pack(fill='x')

#Results Frame
frameResults = tk.Frame(columnOutput)
labelResults = tk.Label(frameResults, text='Results', font=getFont(15))
Hovertip(labelResults, 'See results of the model once it is trained.', 350)
labelResults.pack(pady=(10, 0))

labelOverview = tk.Label(frameResults, text='No model has been trained yet.\n\n\n', font=getFont(12))
labelOverview.pack(fill='x')

#Interact Frame
frameInteract = tk.Frame(columnOutput)
labelInteract = tk.Label(frameInteract, text='Interact', font=getFont(15))
Hovertip(labelInteract, 'Enter data for a trained model to make a prediction.', 350)
labelInteract.pack(pady=(10, 0))

subframeEnterData = tk.Frame(frameInteract)
labelEnterData = tk.Label(subframeEnterData, text='Enter:', font=getFont(11))
stringvarEnterData = StringVar()
stringvarEnterData.set("N/A")
entryEnterData = tk.Entry(subframeEnterData, textvariable=stringvarEnterData)
labelEnterData.pack(side=tk.LEFT, fill='x')
entryEnterData.pack(side=tk.LEFT, fill='x', expand=True)
subframeEnterData.pack(fill='x', padx=20)

buttonInteract = tk.Button(frameInteract, text='Predict', font=getFont(11, 'bold'))
buttonInteract.pack()

labelPrediction = tk.Label(frameInteract, text='Prediction: N/A', font=getFont(11, 'bold'))
Hovertip(labelPrediction, 'Prediction that the AI will make.\nPress the button to make a prediction', 350)
labelPrediction.pack(fill='x')

#Optimize Frame
frameOptimize = tk.Frame(columnOutput)
labelOptimize = tk.Label(frameOptimize, text='Optimization Suggestions', font=getFont(15))
Hovertip(labelOptimize, 'See suggestions to improving the model when trained.', 350)
labelOptimize.pack(pady=(10,0))

subframeSuggestions = tk.Frame(frameOptimize)
labelSuggestions = tk.Label(subframeSuggestions, text='No model has been trained yet.', font=getFont(11))
labelSuggestions.pack(fill='x')
subframeSuggestions.pack(fill='x')

#Save Frame
frameSave = tk.Frame(columnOutput)
labelSave = tk.Label(frameSave, text='Save', font=getFont(15))
Hovertip(labelSave, 'Save the model to use in other Python code\nSave the Instance for to be loaded at a later time.', 350)
labelSave.pack(fill='x')

frameSaveOptions = tk.Frame(frameSave)
buttonSaveModel = tk.Button(frameSaveOptions, text='Save Model', font=getFont(11, 'bold'))
buttonSaveInstance = tk.Button(frameSaveOptions, text='Save Instance', font=getFont(11, 'bold'))
buttonSaveModel.pack(side=tk.LEFT, fill='x', expand=True)
buttonSaveInstance.pack(side=tk.LEFT, fill='x', expand=True)
consistentSize = max(buttonSaveModel.winfo_width(), buttonSaveInstance.winfo_width())
buttonSaveModel.config(width=consistentSize)
buttonSaveInstance.config(width=consistentSize)
frameSaveOptions.pack(fill='x', padx=20)

#PACK UP ALL FRAMES IN OUTPUT COLUMN
frameResults.pack(fill='both', expand=True, pady=5)
frameInteract.pack(fill='both', expand=True, pady=5)
frameOptimize.pack(fill='both', expand=True, pady=5)
frameSave.pack(fill='both', expand=True, pady=5)
#endregion

#PACK UP ALL COLUMNS
columnInput.pack(side=tk.LEFT, fill='both', expand=True)
columnInput.pack_propagate(0)
columnTrain.pack(side=tk.LEFT, fill='both', expand=True)
columnTrain.pack_propagate(0)
columnOutput.pack(side=tk.LEFT, fill='both', expand=True)
columnOutput.pack_propagate(0)

"""
        Additional Loading
"""
#Hide columns that can not be interacted with until data is available
frameSetState(frameInputClean, False)
frameSetState(columnTrain, False)
frameSetState(columnOutput, False)
frameSetState(subframeTargetColumns, False)

def updateGraph():
    global plotWidget, canvasWidget
    plotWidget = figureWidget.clear()
    plotWidget = figureWidget.add_subplot(111)
    plotWidget.plot(Globals.trainedModelGraph)
    plotWidget.set_xlabel("Iterations")
    plotWidget.yaxis.set_label_position("right")
    plotWidget.set_ylabel("Loss")
    plotWidget.set_title("Loss across Iterations")
    canvasWidget.draw()
    
modelSettingsParamenters = {
    "Testing Split": {"type": int, "default": 33},
    "Number of Layers": {"type": int, "default": 100},
    "Number of Neurons": {"type": int, "default": 100},
    "Initial Learning Rate": {"type": float, "default": 0.001},
    "Alpha": {"type": float, "default": 0.0001},
    "Iterations": {"type": int, "default": 200}
}
def parseModelSettings(previousFailed, stringvar, name):
    if(not previousFailed):
        return False, 0
    data = stringvar.get()
    try:
        if(data is None or data == ""): #Blank, use default
            newData = modelSettingsParamenters[name]["default"]
            stringvar.set(newData) #Show user the default
            return True, newData
        
        newDataType = modelSettingsParamenters[name]["type"]
        newData = newDataType(data)
        
        if(newData < 0):
            messagebox.showerror("Invalid data in " + name, "Number cannot be negative!")
            stringvar.set("")
            return False, 0
        
        return True, newData
    except Exception as error:
        if (str(error).find("invalid literal") > -1): #Invalid character (float in int)
            if modelSettingsParamenters[name]["type"] == float:
                messagebox.showerror("Invalid data in " + name, "Number must be a decimal! EX: 1, 1000, 1.0, 0.00001, 2.1234")
            elif modelSettingsParamenters[name]["type"] == int:
                messagebox.showerror("Invalid data in " + name, "Number must be a whole nubmer! EX: 1, 1000, 12345")
            else:
                messagebox.showerror("Invalid data in " + name, "Please change this number so the model can be trained!")
        else:
            messagebox.showerror("Invalid data in " + name, "Please change this number so the model can be trained: " + str(error))
        stringvar.set("")
        return False, 0
    
def getModelSettings():
    dataArray = [0, 0, 0, 0, 0, 0, 0]
    boolValidData = True
    dataArray[0] = stringvarPlatform.get()
    boolValidData, dataArray[1] = parseModelSettings(boolValidData, stringvarNumberOfLayers,"Number of Layers")
    boolValidData, dataArray[2] = parseModelSettings(boolValidData, stringvarNumberOfNeurons, "Number of Neurons")    
    boolValidData, dataArray[3] = parseModelSettings(boolValidData, stringvarInitialLearning, "Initial Learning Rate")
    boolValidData, dataArray[4] = parseModelSettings(boolValidData, stringvarAlpha, "Alpha")  
    boolValidData, dataArray[5] = parseModelSettings(boolValidData, stringvarIterations, "Iterations")
    boolValidData, dataArray[6] = parseModelSettings(boolValidData, intvarSplit, "Testing Split")

    if not boolValidData:
        return []
    else:
        return dataArray

def getCleanSettings():
    return [intvarTextToNumbers.get(), intvarMissingData.get(), intvarResave.get()]  

"""
        Button Functions
"""
def loadDataset():
    global boolCheckFirstRow
    success, newText = Functions.loadDataset()
    
    if(success == 'True'):
        buttonLoadData.config(text=newText)
        intvarFirstRow.set(False)
        boolCheckFirstRow = False
        frameSetState(subframeTargetColumns, True)
        labelTrainingError.config(text='Select targets that the model will predict.\nPress the "Select Columns as Targets"\nbutton to select targets.')
    elif(success == 'Cancel'): #Window was canceled, so continue.
        buttonLoadData.config(text=newText)
        return
    else:
        buttonLoadData.config(text=newText)
        frameSetState(subframeTargetColumns, False)
        labelTrainingError.config(text='No dataset selected.\nPlease selected a dataset.\nHover over text to get useful hints.')
        
    buttonSelectTargetColumns.config(text='None')
    buttonSelectIgnoreColumns.config(text='None')
    buttonLoadInstance.config(text='Select Instance')
    frameSetState(frameInputClean, False)
    frameSetState(columnTrain, False)
    frameSetState(columnOutput, False)
buttonLoadData.config(command=loadDataset)

def setFirstRowData():
    global boolCheckFirstRow
    newFirstRow = intvarFirstRow.get()
    if(boolCheckFirstRow == newFirstRow):
        return
    newTargetText, newIgnoredText = Functions.reformatFirstRow(newFirstRow)
    buttonSelectTargetColumns.config(text=Functions.renameColumnSelectorButtons(newTargetText))
    buttonSelectIgnoreColumns.config(text=Functions.renameColumnSelectorButtons(newIgnoredText))
    boolCheckFirstRow = newFirstRow
radioFirstRowLabel.config(command=setFirstRowData)
radioFirstRowData.config(command=setFirstRowData)

def windowTargetColumnsDone(newColumnList):
    global root, childWindow
    
    if(newColumnList == Globals.targetColumnsList and newColumnList != []): #Nothing was changed
        childWindow.destroy()
        return

    if(Functions.processTargetedColumns(newColumnList)):
        childWindow.destroy()
        buttonSelectTargetColumns.config(text=Functions.renameColumnSelectorButtons(newColumnList))
        buttonSelectIgnoreColumns.config(text=Functions.renameColumnSelectorButtons(Globals.ignoredColumnsList))
        labelTrainingError.config(text='Ready to begin.\nConsider changing paraments in the\nAI model section before training.')
        frameSetState(frameInputClean, True)
        frameSetState(columnTrain, True)
        frameSetState(subframeTargetColumns, True)    
def windowIngoreColumnsDone(newColumnList):
    global root, childWindow
    if(Functions.processIgnoredColumns(newColumnList)):
        childWindow.destroy()
        buttonSelectIgnoreColumns.config(text=Functions.renameColumnSelectorButtons(newColumnList))
def createSelectionWindow(_title, ignoreColumnMode, alreadySelectColumns, finishCommand):
    global iconPath
    """Generic function for making the window to select Ignore and Target Columns"""
    global root, childWindow
    if(ignoreColumnMode and (len(list(Globals.dataset)) - len(Globals.targetColumnsList) <= 1)): #No data left if program ignores column
        messagebox.showerror("Cannot ignore column", "Ignoring any more columns would leave no training data!\nPlease change targets if this is incorrect.")
        return

    childWindow = Toplevel(root)
    childWindow.withdraw()
    childWindow.iconbitmap(iconPath)
    childWindow.title(_title)

    labelWindowText = 'Click on 1 or many rows to select columns.\nClick "Done" to continue.'
    if(ignoreColumnMode):
        labelWindowText = 'Click on 0 or many rows to select columns.\nClick "Done" to continue.'
    labelWindow = tk.Label(childWindow, text=labelWindowText, font=getFont(10, 'bold'))
    labelWindow.pack(fill='x')

    buttonFinished = tk.Button(childWindow, text='Done', command= lambda: finishCommand([listBox.get(i) for i in listBox.curselection()]))
    buttonFinished.pack(fill='x')
    
    scroll = tk.Scrollbar(childWindow)
    scroll.pack(side=tk.RIGHT, fill='y')
    
    listBox = tk.Listbox(childWindow, selectmode='multiple', yscrollcommand=scroll.set)
    listBox.pack(fill='both', expand=True)
    columns = Functions.viewColumns(ignoreColumnMode)
    width = 300
    for col in range(len(columns)):
        listBox.insert(tk.END, columns[col])
        if(len(columns[col]) * 6 > width): #Get the biggest size so the window shows completly
            width = len(columns[col]) * 6
        if(columns[col] in alreadySelectColumns):
            listBox.selection_set(col, col)
    
    childWindow.geometry(f"{width}x{250}+{int(childWindow.winfo_pointerx()/2)}+{int(childWindow.winfo_pointery() / 2)}")
    childWindow.minsize(width, 250)

    childWindow.after(0, childWindow.deiconify)
    childWindow.transient(root)
    childWindow.grab_set()
    root.wait_window(childWindow)
buttonSelectIgnoreColumns.config(command=lambda: createSelectionWindow("Select column(s) to ignore:", True, Globals.ignoredColumnsList, windowIngoreColumnsDone))
buttonSelectTargetColumns.config(command=lambda: createSelectionWindow("Select column(s) to act as targets:", False, Globals.targetColumnsList, windowTargetColumnsDone)) 

currentlyTraining = False
def finishedTraining(success, resultingText, suggestionsText):
    global columnOutput, labelOverview, currentlyTraining
    currentlyTraining = False
    progressBar.stop()
    progressBar.config(mode='determinate')
    labelPrediction.config(text='Prediction: N/A')
    if(success):
        labelTrainingError.config(text="\nModel Trained.\n")
        updateGraph()
        frameSetState(columnOutput, True)
        labelOverview.config(text=resultingText)
        labelSuggestions.config(text=suggestionsText)
        stringvarEnterData.set("")
    else:
        labelTrainingError.config(text=resultingText)
        labelOverview.config(text='No model has been trained yet.\n\n\n')
        labelSuggestions.config(text='No model has been trained yet.')
        stringvarEnterData.set("N/A")
        
def commandBeginTraining():
    global currentlyTraining
    if(currentlyTraining):
        return
    
    progressBar.config(mode='indeterminate')
    progressBar.start()
    frameSetState(columnOutput, False)

    currentlyTraining = True
    Thread(target=lambda: Functions.trainModel(getModelSettings(), getCleanSettings(), finishedTraining)).start()
buttonBeginTraining.config(command=commandBeginTraining)

def interactWithModel():
    global currentlyTraining
    if(currentlyTraining):
        return

    labelPrediction.config(text=Functions.predictWithModel(stringvarEnterData))
buttonInteract.config(command=interactWithModel)

buttonSaveModel.config(command=Functions.saveTrainedModel)
def commandSaveInstance():    
    Functions.saveInstance(labelOverview.cget('text'), labelSuggestions.cget('text'))
buttonSaveInstance.config(command=commandSaveInstance)

def commandLoadInstance():    
    success, modelSettings, cleanSettings, resultsText, suggestionsText = Functions.loadInstance()
    if(success and len(modelSettings) > 1 and len(cleanSettings) > 1):
        updateGraph()    
        stringvarPlatform.set(modelSettings[0])    
        stringvarNumberOfLayers.set(modelSettings[1])
        stringvarNumberOfNeurons.set(modelSettings[2])
        stringvarInitialLearning.set(modelSettings[3])
        stringvarAlpha.set(modelSettings[4])
        stringvarIterations.set(modelSettings[5])
        intvarSplit.set(modelSettings[6])
        
        intvarTextToNumbers.set(cleanSettings[0])
        intvarMissingData.set(cleanSettings[1])
        intvarResave.set(cleanSettings[2])
        
        labelOverview.config(text=resultsText)
        labelSuggestions.config(text=suggestionsText)

        frameSetState(frameInputClean, True)
        frameSetState(columnTrain, True)
        frameSetState(subframeTargetColumns, True)
        frameSetState(columnOutput, True)
        buttonLoadData.config(text='Select File')
        buttonLoadInstance.config(text=Globals.selectedFile)
        labelTrainingError.config(text="\nModel Trained.\n")
        labelPrediction.config(text="Predition: N/A")
        stringvarEnterData.set("")
    buttonSelectTargetColumns.config(text=Functions.renameColumnSelectorButtons(Globals.targetColumnsList))
    buttonSelectIgnoreColumns.config(text=Functions.renameColumnSelectorButtons(Globals.ignoredColumnsList))
buttonLoadInstance.config(command=commandLoadInstance)

if __name__ == "__main__":
    root.mainloop()