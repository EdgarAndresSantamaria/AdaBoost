import nltk
from nltk.tokenize import word_tokenize
from CSVOpener import CSVOpener
from openpyxl import Workbook
import gensim, glob
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA as p
from sklearn.preprocessing import StandardScaler
import Exporter

# Descarga de diccionarios para stop-words
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class Cleaner():

    def __init__(self, dataFilePath, outputFolderPath):
        self.dataFilePath = dataFilePath
        self.outputFolderPath = outputFolderPath

    @staticmethod
    def divideTrainAndDev(dataFile, numTotal):
        file = CSVOpener(dataFile)
        file = file.getFile()

        dataSheet = file["data"]

        # train will be the 60% of the total
        numInstanciasTrain = int(0.60 * numTotal)

        for i in range(numInstanciasTrain+1):
            dataSheet['J%d' % (i+1)] = "train"

        for i in range(numInstanciasTrain+1, numTotal+1):
            dataSheet['J%d' % (i+1)] = "dev"

        file.save(dataFile) # Guardar el nuevo excel

    def tokenize(self, textSucio):
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(textSucio)

        # convert to lower case
        # es decir, convierte la palabra a minúsculas
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        # es decir, quita apóstrofes y posibles acentos
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        # es decir, elimina (, . ? ' [1-9] etc)
        words = [word for word in stripped if word.isalpha()]

        # filter out stop words
        # es decir, filtra preposiciones en inglés
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # stemming of words
        # es decir, guarda unicamente la raiz de la palabra (ej: nurse -> nurs )
        from nltk.stem.porter import PorterStemmer
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]

        cleanTexts = " ".join(words)

        return cleanTexts

    def cleanData(self, contentFile):
        newId = contentFile['A']
        module = contentFile['B']
        site = contentFile['C']
        illnessColumn = contentFile['D']      # array valores clase
        sex = contentFile['E']
        age_years = contentFile['F']
        age_months = contentFile['G']
        age_days = contentFile['H']
        textColumn = contentFile['I']     # array valores texto

        atributos = {'A':newId, 'B':module,'C': site,'D': illnessColumn,
                     'E': sex,'F': age_years,'G': age_months,'H': age_days,
                     'I': textColumn}

        numCorrectInstances = 0

        wLimpio = Workbook() # Crear nuevo excel
        wLimpio.active

        wsa = wLimpio.create_sheet("data", 0)  # crear tab data
        textos = [] # para eliminar repetidos
        for i in range(len(newId)):  # for each row
            write = True

            for letra,atributo in atributos.items():
                # write = False if incorrect data

                valorTexto = atributo[i].value

                if letra == 'F': # is age-years
                    if valorTexto == 999:
                        write = False
                elif letra == 'H': # is age-days
                    if valorTexto == 99:
                        write = False
                elif letra == 'G': # is age-months
                    if valorTexto == 99:
                        write = False
                elif letra == 'E': # is sex
                    if valorTexto == 9:
                        write = False
                elif letra == 'I':  # is texto
                    if valorTexto is not None:
                        # evitar repetidos
                        limpio = self.tokenize(valorTexto)
                        if limpio in textos or limpio == "commentsno comment" or limpio == "comment" or limpio=="nonenon":
                            write = False
                        else:
                            textos.append(limpio)
                    else:
                        write = False

                if write is False:
                    break

            if write is True: # if correct data
                numCorrectInstances+=1
                for letra,atributo in atributos.items():
                    valorTexto = atributo[i].value
                    if letra == 'I':
                        valorTexto = self.tokenize(valorTexto)

                    wsa[letra+'%d' % numCorrectInstances] = valorTexto

        wLimpio.save(self.outputFolderPath + "/1datosLimpios.xlsx") # Guardar el nuevo excel

    @staticmethod
    def getDataFileContent(dataFilePath):
        opener = CSVOpener(dataFilePath)
        contentFile = opener.getFileContent("data")
        return contentFile

    def main(self):
        print("Se cargan los datos")
        contenidoFichero = Cleaner.getDataFileContent(self.dataFilePath)
        print("Se empieza a limpiar el fichero")
        self.cleanData(contenidoFichero)
        print("Los ficheros se han limpiado")

class Doc2Vec:

    def __init__(self, outputFolder):
        self.dataFile = outputFolder + "/1datosLimpios.xlsx"
        self.outputFolder = outputFolder

    def trainDoc2Vec(self, vec_size, max_epochs, num_models, learning_rate):
        print("Se entrena el Doc2Vec")

        fileOpener = CSVOpener(self.dataFile)
        data = fileOpener.getDataText("data") # ["texto1","texto2",...]

        tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(data)]

        #alpha = 0.025 por defecto
        alpha = float(learning_rate)

        for i in range(int(num_models)):
            print("Entrenando modelo %d" % (i+1))
            model = gensim.models.doc2vec.Doc2Vec(vector_size=int(vec_size), window=5, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1, workers=24)

            model.build_vocab(tagged_data)

            for epoch in range(int(max_epochs)):
                print("epoch %d/%s" % ((int(epoch)+1),max_epochs))
                model.train(tagged_data,
                            total_examples=model.corpus_count,
                            epochs=model.iter)
                # decrease the learning rate
                model.alpha -= 0.0002
                # fix the learning rate, no decay
                model.min_alpha = model.alpha

            model.save(self.outputFolder + "/2modelo%d.model" % (i+1))
            print("Modelo %d guardado" % (i+1))

    @staticmethod
    def loadModel(modelFile):
        model= gensim.models.doc2vec.Doc2Vec.load(modelFile)
        return model

class PCA:

    def __init__(self,outputFolder):
        self.OUTPUT_FOLDER = outputFolder

    def mostRepresentative2Attributes(self):
        #generamos una lista con los dos vectores mas representativos
        #pca = PCA(n_components=2)
        #X = self.model[self.model.wv.vocab]
        #result = pca.fit_transform(X)
        #self.pca=result
        pass

    def reduceDimension(self):
        print("Reducing dimension of the vectors")

        self.MODELS = glob.glob(self.OUTPUT_FOLDER+'/2modelo*.model') # models to use in each iteration

        i = 1
        for modelFile in self.MODELS:
            print("%d/%d" % (i, len(self.MODELS)))

            # Load Model
            oneModel = Doc2Vec.loadModel(modelFile)
            vectors = oneModel.docvecs.vectors_docs

            # reduce dimension of the vectors
            pca = p(n_components=2) #mantain the 85% of the variance
            X_normalized = StandardScaler().fit_transform(vectors)
            X_reduced = pca.fit_transform(X_normalized)

            # get the first 1000 vectors
            X_reduced = X_reduced[:100]

            # Save the new vectors
            e = Exporter.Exporter(self.OUTPUT_FOLDER + "/3VectorsPCA%d.sav" % i)
            e.save_data(X_reduced)

            print("Saved")
            i+=1

        # Divide train and dev
        # all data
        #Cleaner.divideTrainAndDev(self.OUTPUT_FOLDER + "/1datosLimpios.xlsx", len(X_reduced))
        # the first 1000
        Cleaner.divideTrainAndDev(self.OUTPUT_FOLDER + "/1datosLimpios.xlsx", 100)
