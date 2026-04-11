# -*- coding: utf-8 -*-
"""
Autor: Gabriel Gutiérrez y Marcos Cobo.
Script para la implementación del test.
"""

import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import json
import os
from colorama import Fore
# Sklearn
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)",
                       required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,
                       type=int)
    parse.add_argument("--debug",
                       help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]",
                       required=False, default=False, action="store_true")
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
        # Comprobamos si la columna especificada en --prediction está en el CSV
        if args.prediction in data.columns:
            data.drop(columns=[args.prediction], inplace=True)
            print(
                Fore.YELLOW + f"Columna de predicción '{args.prediction}' encontrada y eliminada del dataset." + Fore.RESET)
        else:
            print(
                Fore.CYAN + f"La columna '{args.prediction}' no se encontró en el archivo, procediendo normalmente." + Fore.RESET)
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)

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
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[
            :, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Columnas numéricas:\n" + Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA + "> Columnas de texto:\n" + Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA + "> Columnas categóricas:\n" + Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
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
            print(Fore.GREEN + "Missing values eliminados con éxito" + Fore.RESET)
        elif args.preprocessing["missing_values"] == "impute":
            if args.preprocessing["impute_strategy"] == "mean":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(
                    data[numerical_feature.columns].mean())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(
                    data[categorical_feature.columns].mean())
                print(Fore.GREEN + "Missing values imputados con éxito usando la media" + Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "median":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(
                    data[numerical_feature.columns].median())
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(
                    data[categorical_feature.columns].median())
                print(Fore.GREEN + "Missing values imputados con éxito usando la mediana" + Fore.RESET)
            elif args.preprocessing["impute_strategy"] == "most_frequent":
                data[numerical_feature.columns] = data[numerical_feature.columns].fillna(
                    data[numerical_feature.columns].mode().iloc[0])
                data[categorical_feature.columns] = data[categorical_feature.columns].fillna(
                    data[categorical_feature.columns].mode().iloc[0])
                print(Fore.GREEN + "Missing values imputados con éxito usando la moda" + Fore.RESET)
            else:
                print(Fore.GREEN + "No se ha seleccionado ninguna estrategia de imputación" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se están tratando los missing values" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar los missing values" + Fore.RESET)
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
                print(Fore.GREEN + "Datos reescalados con éxito usando MinMaxScaler" + Fore.RESET)
            elif args.preprocessing["scaling"] == "normalizer":
                scaler = Normalizer()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados con éxito usando Normalizer" + Fore.RESET)
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados con éxito usando MaxAbsScaler" + Fore.RESET)
            elif args.preprocessing["scaling"] == "standard":
                scaler = StandardScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados con éxito usando StandardScaler" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están escalando los datos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas numéricas" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
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
            print(Fore.GREEN + "Datos categóricos pasados a numéricos con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas categóricas que pasar a numericas" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al pasar los datos categóricos a numéricos" + Fore.RESET)
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
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            for col in text_feature.columns:
                data[col] = data[col].apply(lambda x: ' '.join(sorted(
                    [stemmer.stem(word) for word in word_tokenize(x.lower()) if
                     word not in stop_words and word not in string.punctuation])))
            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a simplificar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
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
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(),
                                                columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

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

    # Realizamos Oversampling o Undersampling
    print(Fore.GREEN + "No se realiza oversampling o undersampling en modo test" + Fore.RESET)

    drop_features()

    return data

def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)

    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Test ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN + "La carpeta output ya existe" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
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
            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)

    # Cargamos el modelo
    print("\n- Cargando modelo...")
    model = load_model()
    # Predecimos
    print("\n- Prediciendo...")
    try:
        predict()
        print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)
        # Guardamos el dataframe con la prediccion
        data.to_csv('output/data-prediction.csv', index=False)
        print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
