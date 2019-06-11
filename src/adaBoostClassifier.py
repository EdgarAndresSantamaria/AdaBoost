from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from CSVOpener import CSVOpener
from sklearn.feature_extraction.text import TfidfVectorizer

class adaBoostClassifier:
    def __init__(self, outputfolder):
        self.outputFolder = outputfolder



    def trainAdaBoostClassifierTFIDF(self):

        print("Se aplica TF-IDF")
        for fichero in listdir(self.outputFolder):
            c = CSVOpener(self.outputFolder+"/"+fichero)

        data = c.getDataText("data")  # ["texto1","texto2",...]
        sheet = c.getFileContent("data")

        # clases a predecir
        inicial = True
        Y = []
        for cell in sheet['D']:
            if (not inicial):
                Y.append(cell.value)  # just the text
            else:
                inicial = False


        n_split = int(0.9*len(data))

        print(len(data))

        # aplicar representacion TF-IDF
        vectorizer = TfidfVectorizer(norm='l1',sublinear_tf=True)
        X = vectorizer.fit_transform(data[:n_split])
        # sparse to non sparse
        X_train = X.toarray()
        X = vectorizer.transform(data[n_split:])
        # sparse to non sparse
        X_test = X.toarray()

        # reduccion de dimensionalidad en funcion de la varianza (ruido)
        # mas lento...
        pca = PCA(n_components=96)

        # reduccion de dimensionalidad

        # mas rapido...
        #pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print("conjunto preprocesado")

        y_train, y_test = Y[:n_split], Y[n_split:]
        Y = np.concatenate((y_train, y_test), axis=0)
        X = np.concatenate((X_train, X_test), axis=0)

        #valores a combinar baseline
        param_dist = {
            'n_estimators': [10],
            #'n_estimators': [100],
            'learning_rate': [0.1],
            'base_estimator': [RandomForestClassifier(n_estimators=100,bootstrap=True,max_features=96,max_depth=96)]
        }

        #busqueda de valores optimos, prueba todas las combinaciones posibles

        pre_gs_inst = GridSearchCV(AdaBoostClassifier(),
                                         param_grid=param_dist,
                                         cv=3,scoring='accuracy',
                                         n_jobs=16,verbose=2)

        pre_gs_inst.fit(X, Y)

        print("el mejor estimador es:")
        print(pre_gs_inst.best_estimator_)
        print("su evaluacion es")
        print(pre_gs_inst.best_score_)

        print("evaluacion de valores óptimos del algoritmo")
        print("evaluando algoritmo real")

        #validacion del sistema con valores optimos
        #probas = cross_val_predict(pre_gs_inst.best_estimator_, X_train, y_train, cv=StratifiedKFold(n_splits=100, random_state=8),
        #                           n_jobs=6, method='predict_proba', verbose=2)

        bdt_real=pre_gs_inst.best_estimator_
        #bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,min_samples_leaf=7), learning_rate=0.2, n_estimators=200)
        probas = cross_val_predict(bdt_real,X ,Y, cv=StratifiedKFold(n_splits=3, random_state=8),n_jobs=3, method='predict_proba', verbose=2)

        pred_indices = np.argmax(probas, axis=1)
        classes = np.unique(Y)
        preds = classes[pred_indices]

        #display de resultados
        print('Log loss: {}'.format(log_loss(Y, probas, labels=classes)))
        print('Accuracy: {}'.format(accuracy_score(Y, preds)))
        print('f-score: {}'.format(f1_score(Y, preds, average="macro", labels=classes)))
        print('Precision: {}'.format(precision_score(Y, preds, pos_label=1, average='macro')))


        print("evaluando algoritmo discreto")

        bdt_discrete = bdt_real
        probas1 = cross_val_predict(bdt_discrete, X, Y, cv=StratifiedKFold(n_splits=3, random_state=0), n_jobs=3,method='predict_proba', verbose=2)
        pred_indices1 = np.argmax(probas1, axis=1)
        classes1 = np.unique(Y)
        preds1 = classes1[pred_indices1]

        # display de resultados
        print('Log Loss: {}'.format(log_loss(Y, probas1)))
        print('Accuracy: {}'.format(accuracy_score(Y, preds1)))
        print('f-score: {}'.format(f1_score(Y, preds1, average="macro")))
        print('Precision: {}'.format(precision_score(Y, preds1, pos_label=1, average='macro')))



        #print("entrenando modelo real  (optimizado)...")
        bdt_real.fit(X_train, y_train)

        #print("entrenando modelo discreto  (no optimizado)...")
        bdt_discrete.fit(X_train, y_train)

        real_test_errors = []
        discrete_test_errors = []
        print("calculando errores...")
        avg=0
        for real_test_predict in bdt_real.staged_predict(X_test):
            real_test_errors.append(1 - accuracy_score(real_test_predict, y_test))
            avg+=1 - accuracy_score(real_test_predict, y_test)

        for discrete_train_predict in bdt_discrete.staged_predict(X_test):
            discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))



        n_trees_discrete = len(bdt_discrete)
        n_trees_real = len(bdt_real)
        avg = avg / n_trees_real

        #Boosting might terminate early, but the following arrays are always
        #n_estimators long. We crop them to the actual number of trees here:
        discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
        real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
        real_estimator_weights = bdt_real.estimator_weights_[:n_trees_real]
        discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors,
                 "b", label='SAMME.R overfitted', alpha=.5)
        plt.plot(range(1, n_trees_real + 1), real_test_errors,
                 "r", label='SAMME.R', alpha=.5)
        plt.legend()
        plt.ylim(-0.5, 1.5)
        plt.xlim((-20, len(bdt_real) + 20))
        plt.ylabel('Test Error (avg '+ str(avg)+")")
        plt.xlabel('Number of Trees')


        plt.subplot(132)  
        plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
                 "b", label='SAMME.R overfitted', alpha=.5)
        plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
                 "r", label='SAMME.R', alpha=.5)
        plt.legend()
        plt.ylabel('Train Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                 max(real_estimator_errors.max(),
                     discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_real) + 20))

        plt.subplot(133)
        
        plt.plot(range(1, n_trees_real + 1), real_estimator_weights,
                 "b", label='SAMME-R')
        plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
                 "r", label='SAMME.R overfitted')
        plt.legend()
        plt.ylabel('Weight')
        plt.xlabel('Number of Trees')
        plt.ylim((0, real_estimator_weights.max() * 1.2))
        plt.xlim((-20, n_trees_real + 20))

        # prevent overlapping y-axis labels
        plt.subplots_adjust(wspace=0.5)
        plt.show()

