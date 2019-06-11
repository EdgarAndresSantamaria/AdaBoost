import openpyxl


class CSVOpener():

    def __init__(self, pathDataFile):
        self.path = pathDataFile

    def getFile(self):
        wb = openpyxl.load_workbook(self.path)    #cargar archivo (.xlsx)
        return wb

    def getFileContent(self,sheetName):
        wb = self.getFile()
        dataSheet = wb[sheetName]     # Seleccionar tab
        return dataSheet

    def getDataText(self,sheetName):
        dataSheet = self.getFileContent(sheetName)

        datatext = []

        textColumn = dataSheet['I']

        for i in range(1,len(textColumn)):
        #for i in range(1,50):
            text = str(textColumn[i].value)
            datatext.append(text)

        return datatext
