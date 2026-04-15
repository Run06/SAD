# -*- coding: utf-8 -*-
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
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm


def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    parse = argparse.ArgumentParser(description="Algoritmo de clasificación con reporte detallado.")
    parse.add_argument("-f", "--file", help="Fichero csv", required=True)
    parse.add_argument("-a", "--algorithm", help="kNN, decision_tree o random_forest", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir", required=True)
    parse.add_argument("-e", "--estimator", help="Scoring (f1_macro, accuracy...)", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="CPUs", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra métricas", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")

    args = parse.parse_args()
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def load_data(file):
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + f"Error al cargar: {e}" + Fore.RESET);
        sys.exit(1)


# --- Preprocesamiento Corregido ---

def preprocesar_datos():
    global data
    # 1. Separar tipos
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if args.prediction in numerical_cols: numerical_cols.remove(args.prediction)

    categorical_all = data.select_dtypes(include='object').columns.tolist()
    if args.prediction in categorical_all: categorical_all.remove(args.prediction)

    cat_cols = [c for c in categorical_all if data[c].nunique() <= args.preprocessing["unique_category_threshold"]]
    text_cols = [c for c in categorical_all if c not in cat_cols]

    # 2. Simplificar texto (Soporte Español/Inglés)
    if text_cols:
        lang = args.preprocessing.get("language", "english")
        stop_words = set(stopwords.words(lang))
        stemmer = SnowballStemmer(lang)
        for col in text_cols:
            data[col] = data[col].astype(str).apply(lambda x: ' '.join(sorted([
                stemmer.stem(w) for w in word_tokenize(x.lower())
                if w not in stop_words and w not in string.punctuation
            ])))

    # 3. Categorical a Numerical (LabelEncoding por columna)
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # 4. Missing Values (Corrección lógica: media solo a números)
    strat = args.preprocessing["missing_values"]
    if strat == "drop":
        data = data.dropna(subset=numerical_cols + cat_cols).reset_index(drop=True)
    elif strat == "impute":
        imp = args.preprocessing["impute_strategy"]
        for col in numerical_cols:
            val = data[col].mean() if imp == "mean" else data[col].median()
            data[col] = data[col].fillna(val)
        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

    # 5. Scaling
    if numerical_cols:
        sc_type = args.preprocessing["scaling"]
        scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "maxabs": MaxAbsScaler(),
                   "normalizer": Normalizer()}
        if sc_type in scalers:
            data[numerical_cols] = scalers[sc_type].fit_transform(data[numerical_cols])

    # 6. Vectorización de Texto
    if text_cols:
        method = args.preprocessing["text_process"]
        vec = TfidfVectorizer() if method == "tf-idf" else CountVectorizer()
        combined_text = data[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        matrix = vec.fit_transform(combined_text)
        text_df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())
        data.reset_index(drop=True, inplace=True)
        data = pd.concat([data, text_df], axis=1).drop(columns=text_cols)

    # 7. Drop features manuales
    if args.preprocessing["drop_features"]:
        data.drop(columns=args.preprocessing["drop_features"], inplace=True, errors='ignore')


# --- Guardado de Resultados (Exacto a tu imagen) ---

def save_detailed_report(gs, name, x_dev, y_dev):
    try:
        # Guardar el modelo físico
        with open(f'output/modelo_{name}.pkl', 'wb') as f:
            pickle.dump(gs, f)

        # Calcular métricas finales
        y_pred = gs.predict(x_dev)
        f1_mic = f1_score(y_dev, y_pred, average='micro')
        f1_mac = f1_score(y_dev, y_pred, average='macro')

        # Crear el CSV con la estructura que pediste
        csv_path = f'output/resultado_{name}.csv'
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Sección: Resumen
            writer.writerow(['--- RESUMEN FINAL ---'])
            writer.writerow(['Metrica', 'Valor'])
            writer.writerow(['Mejor Score (CV)', gs.best_score_])
            writer.writerow(['F1 Micro (Dev)', f1_mic])
            writer.writerow(['F1 Macro (Dev)', f1_mac])
            writer.writerow(['<null>'])  # Espacio como en tu imagen

            # Sección: Historial detallado de todas las pruebas
            writer.writerow(['--- HISTORIAL DE GRID SEARCH ---'])
            writer.writerow(['Parametros', 'Media Score CV'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])

        print(Fore.CYAN + f"\nReporte detallado generado en: {csv_path}" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Error al guardar reporte: {e}" + Fore.RESET)


# --- Ejecución ---

def run_process(model, params, name):
    # Dividir datos
    y = data[args.prediction]
    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
    x = data.drop(columns=[args.prediction])

    xt, xd, yt, yd = train_test_split(x, y, test_size=0.25, random_state=42)

    # Sampling
    if args.preprocessing["sampling"] == "oversampling":
        xt, yt = RandomOverSampler(random_state=42).fit_resample(xt, yt)
    elif args.preprocessing["sampling"] == "undersampling":
        xt, yt = RandomUnderSampler(random_state=42).fit_resample(xt, yt)

    # Entrenamiento con barra de progreso
    with tqdm(total=100, desc=f'Procesando {name}', unit='iter') as pbar:
        gs = GridSearchCV(model, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start = time.time()
        gs.fit(xt, yt)
        end = time.time()
        pbar.update(100)

    print(f"Tiempo de ejecución: {Fore.MAGENTA}{end - start:.2f}{Fore.RESET} segundos")

    if args.verbose:
        print(Fore.YELLOW + "Classification Report:" + Fore.RESET)
        print(classification_report(yd, gs.predict(xd), zero_division=0))

    save_detailed_report(gs, name, xd, yd)


if __name__ == "__main__":
    np.random.seed(42)
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    if not os.path.exists('output'): os.makedirs('output')

    print("- Cargando datos...")
    data = load_data(args.file)

    print("- Descargando diccionarios...")
    nltk.download(['stopwords', 'punkt', 'punkt_tab'], quiet=True)

    print("- Preprocesando datos...")
    preprocesar_datos()

    if args.debug:
        data.to_csv('output/data-processed.csv', index=False)
        print(Fore.GREEN + "CSV de debug guardado." + Fore.RESET)

    print("- Ejecutando algoritmo...")
    if args.algorithm == "kNN":
        p = args.kNN
        # Corregimos el rango de vecinos [inicio, fin, salto] -> lista
        ks = list(range(p['n_neighbors'][0], p['n_neighbors'][1] + 1, p['n_neighbors'][2]))
        p['n_neighbors'] = ks
        run_process(KNeighborsClassifier(), p, "kNN")
    elif args.algorithm == "decision_tree":
        run_process(DecisionTreeClassifier(), args.decision_tree, "decision_tree")
    elif args.algorithm == "random_forest":
        run_process(RandomForestClassifier(), args.random_forest, "random_forest")
    else:
        print(Fore.RED + "Algoritmo no soportado." + Fore.RESET)