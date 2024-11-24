datasetPath = ""
selectedFile = "Select File"

dataset = None
originalColumnNames = []
originalRowNumber = 0

targetColumnsList = []
ignoredColumnsList = []

targetColumnsIndex = []
rawInputsColumnsIndex = []
finalInputsColumnsIndex = []
ignoredColumnsIndex = []

modelUsedModelSettings = []
modelUsedCleanSettings = []

def resetArrays():
    global targetColumnsList, ignoredColumnsList, targetColumnsIndex, rawInputsColumnsIndex, finalInputsColumnsIndex, ignoredColumnsIndex
    targetColumnsList = []
    ignoredColumnsList = []

    targetColumnsIndex = []
    rawInputsColumnsIndex = []
    finalInputsColumnsIndex = []
    ignoredColumnsIndex = []

trainedModel = None
trainedModelGraph = None