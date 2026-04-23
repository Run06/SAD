# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munne.
Script para la implementación del entrenamiento de Análisis de Sentimientos.
"""

import sys, argparse, pandas as pd, numpy as np, string, pickle, json, os
from colorama import Fore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def parse_args():
    parser = argparse.ArgumentParser(description="PASO 1: Entrenamiento de Análisis de Sentimientos")
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-a", "--algorithm", required=True, choices=["kNN", "decision_tree", "random_forest", "logistic_regression"])
    parser.add_argument("-p", "--prediction", required=True, help="Columna objetivo (ej. score)")
    parser.add_argument("-e", "--estimator", default="f1_macro")
    parser.add_argument("-c", "--cpu", default=-1, type=int)
    parser.add_argument("-s", "--sample", type=float, default=1.0, help="Porcentaje de datos a usar (0.0 a 1.0)")

    args = parser.parse_args()

    # Cargar configuración adicional desde JSON
    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args


def preprocesar_entrenamiento(data, args):
    # 1. Mapeo de Score a Sentimiento
    # 1-2: Negativo, 3: Neutral, 4-5: Positivo
    mapeo = {1: 'negativo', 2: 'negativo', 3: 'neutral', 4: 'positivo', 5: 'positivo'}
    data[args.prediction] = data[args.prediction].map(mapeo)

    # 2. Identificación de columnas
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if args.prediction in num_cols: num_cols.remove(args.prediction)

    cat_all = data.select_dtypes(include=['object', 'string']).columns.tolist()
    if args.prediction in cat_all: cat_all.remove(args.prediction)

    # Umbral para decidir si una columna es categoría simple o texto largo
    threshold = args.preprocessing.get("unique_category_threshold", 10)
    cat_cols = [c for c in cat_all if data[c].nunique() <= threshold]
    text_cols = [c for c in cat_all if c not in cat_cols]

    # 3. Limpieza de Texto (Stemming y Stopwords)
    lang = args.preprocessing.get("spanish", "english")
    stemmer = SnowballStemmer(lang)
    stop_words = set(stopwords.words(lang))

    def clean_text(text):
        tokens = word_tokenize(str(text).lower())
        # Filtramos: es alfanumérico, no es stopword y no es puntuación
        cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and w not in string.punctuation]
        return ' '.join(cleaned)

    for col in text_cols:
        data[col] = data[col].apply(clean_text)

    # 4. Vectorización de Texto
    if args.preprocessing["text_process"] == "tf-idf":
        vec = TfidfVectorizer(max_features=2000)  # Limitamos para evitar colapso de RAM
    else:
        vec = CountVectorizer(max_features=2000)

    # Unimos todas las columnas de texto en una sola representación
    text_data = data[text_cols].apply(lambda x: ' '.join(x), axis=1)
    txt_mat = vec.fit_transform(text_data)

    # 5. Escalado de numéricas
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "maxabs": MaxAbsScaler(),
        "normalizer": Normalizer()
    }
    scaler = scalers.get(args.preprocessing["scaling"], StandardScaler())

    if num_cols:
        data[num_cols] = scaler.fit_transform(data[num_cols].fillna(0))

    # 6. Codificación de la variable objetivo
    le = LabelEncoder()
    y = le.fit_transform(data[args.prediction])

    # 7. Combinación final (Numéricas + Texto Vectorizado)
    X_num = data[num_cols].reset_index(drop=True)
    X_txt = pd.DataFrame(txt_mat.toarray(), columns=vec.get_feature_names_out())
    X = pd.concat([X_num, X_txt], axis=1)

    return X, y, vec, scaler, le, text_cols, num_cols


def run_train(X, y, model, params, name, vec, scaler, le, text_cols, num_cols, args):
    # Split
    xt, xd, yt, yd = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Guardar set de validación
    dev_df = pd.concat([xd.reset_index(drop=True), pd.Series(yd, name=args.prediction)], axis=1)
    dev_df.to_csv('output/traindev/dev_set.csv', index=False)

    # Balanceo de carga
    sampling = args.preprocessing.get("sampling", "none")
    if sampling == "oversampling":
        xt, yt = RandomOverSampler().fit_resample(xt, yt)
    elif sampling == "undersampling":
        xt, yt = RandomUnderSampler().fit_resample(xt, yt)

    # Entrenamiento con búsqueda de hiperparámetros
    gs = GridSearchCV(model, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    gs.fit(xt, yt)

    # Guardar el artefacto completo
    pipeline = {
        'modelo': gs.best_estimator_,
        'params': gs.best_params_,
        'vec': vec,
        'scaler': scaler,
        'le': le,
        'text_cols': text_cols,
        'num_cols': num_cols
    }

    with open(f'output/traindev/modelo_{name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print(Fore.CYAN + f"Modelo {name} entrenado con resultado aproximado {args.estimator}: {gs.best_score_:.4f}" + Fore.RESET)
    print(Fore.CYAN + f"Ejecuta el dev para ver los resultados reales obtenidos" + Fore.RESET)
    print(Fore.GREEN + f"Guardado en: output/traindev/modelo_{name}.pkl" + Fore.RESET)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists('output/traindev'): os.makedirs('output/traindev')

    # Descargas necesarias
    nltk.download(['stopwords', 'punkt', 'punkt_tab'], quiet=True)

    # Carga de datos
    full_data = pd.read_csv(args.file,usecols=['review','score'])

    if args.sample < 1.0:
        # random_state seed (semilla)
        data = full_data.sample(frac=args.sample, random_state=42).reset_index(drop=True)
        print(
            Fore.YELLOW + f"Trabajando con una muestra del {args.sample * 100}% ({len(data)} registros)." + Fore.RESET)
    else:
        data = full_data
        print(Fore.GREEN + f"Trabajando con el dataset completo ({len(data)} registros)." + Fore.RESET)

    # Preprocesamiento
    X, y, vec, scaler, le, text_cols, num_cols = preprocesar_entrenamiento(data, args)

    # Selección de algoritmo
    if args.algorithm == "kNN":
        p = args.kNN
        # Asegurar que n_neighbors sea una lista min, max step
        if isinstance(p['n_neighbors'], list) and len(p['n_neighbors']) == 3:
            p['n_neighbors'] = list(range(p['n_neighbors'][0], p['n_neighbors'][1] + 1, p['n_neighbors'][2]))
        run_train(X, y, KNeighborsClassifier(), p, "kNN", vec, scaler, le, text_cols, num_cols, args)

    elif args.algorithm == "decision_tree":
        run_train(X, y, DecisionTreeClassifier(), args.decision_tree, "decision_tree", vec, scaler, le, text_cols,
                  num_cols, args)

    elif args.algorithm == "random_forest":
        run_train(X, y, RandomForestClassifier(), args.random_forest, "random_forest", vec, scaler, le, text_cols,
                  num_cols, args)

    elif args.algorithm == "logistic_regression":
        run_train(X, y, LogisticRegression(max_iter=1000), args.logistic_regression, "logistic_regression",
                  vec, scaler, le, text_cols, num_cols, args)

    else:
        print("Algoritmo no soportado.")
        sys.exit(-1)