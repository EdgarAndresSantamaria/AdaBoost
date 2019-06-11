import sys
from Preprocesador import Cleaner, Doc2Vec, PCA
from adaBoostClassifier import adaBoostClassifier
import Utilities


inputArguments = sys.argv

'''
nota importante, el sistema funciona en pipeline de tal manera que los ficheros parciales de los
estadios anteriores alimentan los siguientes estadios con este orden: -p -> -h -> -a -> -c
'''
# -p -i datosSucios.xlsx vec_size max_epochs learning_Rate -o carpetaOutput   // Preprocesador red neuronal
# -p -t datosSucios.xlsx -o carpetaOutput   // adaBoost + TFIDF
#

done = False
if (len(inputArguments)-1) == 8: # Cleaner
    if inputArguments[1] == "-p" and inputArguments[2] == "-i" and inputArguments[7] == "-o":
        Utilities.createFolderIfNotExists(inputArguments[8])
        cleaner = Cleaner(inputArguments[3],inputArguments[8])
        cleaner.main()
        modelo = Doc2Vec(inputArguments[8])
        modelo.trainDoc2Vec(vec_size=inputArguments[4],max_epochs=inputArguments[5],num_models=1,learning_rate=inputArguments[6])
        pca = PCA(inputArguments[8])
        pca.reduceDimension()
        done = True
    else:
        print("error en argumentos")

if (len(inputArguments)-1) == 5: # Cleaner
    if inputArguments[1] == "-p" and inputArguments[2] == "-t" and inputArguments[4] == "-o":
        Utilities.createFolderIfNotExists(inputArguments[5])
        #cleaner = Cleaner(inputArguments[3],inputArguments[5])
        #cleaner.main()
        modelo = adaBoostClassifier(inputArguments[5])
        modelo.trainAdaBoostClassifierTFIDF()
        done = True
    else:
        print("error en argumentos")
