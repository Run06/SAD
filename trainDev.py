# -*- coding: utf-8 -*-
"""
Autor: Gabriel Gutiérrez y Marcos Cobo.
Script para la implementación del train-dev.
"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# Funciones auxiliares

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos t0do en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para calcular métricas

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_classification_report(y_test, y_pred):
    """
    Función para calcular el informe de clasificación
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Informe de clasificación
    """
    report = classification_report(y_test, y_pred, zero_division=0)
    return report

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        if args.preprocessing["missing_values"] == "drop":
            data = data.dropna(subset=numerical_feature.columns)
            data = data.dropna(subset=categorical_feature.columns)
            print(Fore.GREEN+"Missing values eliminados con éxito"+Fore.RESET)
        elif args.preprocessing["missing_values"] == "impute":
            if args.preprocessing["impute_strategy"] == "mean":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mean())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mean())
                print(Fore.GREEN+"Missing values imputados con éxito usando la media"+Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "median":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].median())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].median())
                print(Fore.GREEN+"Missing values imputados con éxito usando la mediana"+Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "most_frequent":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(data[numerical_feature.columns].mode().iloc[0])
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(data[categorical_feature.columns].mode().iloc[0])
                print(Fore.GREEN+"Missing values imputados con éxito usando la moda"+Fore.RESET)
            else:
                print(Fore.GREEN+"No se ha seleccionado ninguna estrategia de imputación"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se están tratando los missing values"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar los missing values"+Fore.RESET)
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    global data
    try:
        if numerical_feature.columns.size > 0:
            if args.preprocessing["scaling"] == "minmax":
                scaler = MinMaxScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando MinMaxScaler"+Fore.RESET)
            elif args.preprocessing["scaling"] == "normalizer":
                scaler = Normalizer()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando Normalizer"+Fore.RESET)
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando MaxAbsScaler"+Fore.RESET)
            elif args.preprocessing["scaling"] == "standard":
                scaler = StandardScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN+"Datos reescalados con éxito usando StandardScaler"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están escalando los datos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas numéricas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al reescalar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global data
    try:
        if categorical_feature.columns.size > 0:
            labelencoder = LabelEncoder()
            for col in categorical_feature.columns:
                data[col] = labelencoder.fit_transform(data[col])
            print(Fore.GREEN+"Datos categóricos pasados a numéricos con éxito"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas categóricas que pasar a numericas"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al pasar los datos categóricos a numéricos"+Fore.RESET)
        print(e)
        sys.exit(1)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame.
    
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    
    Retorna:
    None
    """
    global data
    try:
        if text_feature.columns.size > 0:
            stop_words = set(stopwords.words('english')) #spanish
            stemmer = PorterStemmer() #SnowballStemmer
            for col in text_feature.columns:
                data[col] = data[col].apply(lambda x: ' '.join(sorted([stemmer.stem(word) for word in word_tokenize(x.lower()) if word not in stop_words and word not in string.punctuation])))
            print(Fore.GREEN+"Texto simplificado con éxito"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a simplificar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al simplificar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":               
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling(x_train, y_train):
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    """

    try:
        if args.preprocessing["sampling"] == "oversampling":
            ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
            x_train, y_train = ros.fit_resample(x_train, y_train)
            print(Fore.BLUE + "Oversampling aplicado solo al TRAIN" + Fore.RESET)

        elif args.preprocessing["sampling"] == "undersampling":
            rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            x_train, y_train = rus.fit_resample(x_train, y_train)
            print(Fore.BLUE + "Undersampling aplicado solo al TRAIN" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error en el sampling del entrenamiento: " + str(e) + Fore.RESET)

    return x_train, y_train

def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos 
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        TODO 5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Simplificamos el texto
    simplify_text(text_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    
    # Tratamos el texto
    process_text(text_feature)

    drop_features()

    return data

# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    # Sacamos la columna a predecir
    y = data[args.prediction]
    x = data.drop(columns=[args.prediction])
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.25, random_state=42)

    #Aplicar over y undersamplig a la muestra train
    x_train, y_train = over_under_sampling(x_train, y_train)

    return x_train, x_dev, y_train, y_dev

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        param_grid = args.kNN 
        min_max_step = param_grid['n_neighbors']
        i = min_max_step[0]
        ks = []
        while i <= min_max_step[1]:
            ks.append(i)
            i = i + min_max_step[2]
        param_grid['n_neighbors'] = ks
        gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(DecisionTreeClassifier(), args.decision_tree, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)


# Función principal
if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Train-Dev ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    # Ejecutamos el algoritmo seleccionado
    print("\n- Ejecutando algoritmo...")
    if args.algorithm == "kNN":
        try:
            kNN()
            print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "decision_tree":
        try:
            decision_tree()
            print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "random_forest":
        try:
            random_forest()
            print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    else:
        print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
        sys.exit(1)
