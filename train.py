# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munné.
Script para la implementación del entrenamiento de Análisis de Sentimientos.
"""

import sys, argparse, pandas as pd, numpy as np, string, pickle, json, os, re
from colorama import Fore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
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
from ollama import chat


def parse_args():
    parser = argparse.ArgumentParser(description="PASO 1: Entrenamiento de Análisis de Sentimientos")
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-a", "--algorithm", required=True,
                        choices=["kNN", "decision_tree", "random_forest", "logistic_regression", "naive_bayes"])
    parser.add_argument("-p", "--prediction", required=True, help="Columna objetivo (ej. score)")
    parser.add_argument("-e", "--estimator", default="f1_macro")
    parser.add_argument("-c", "--cpu", default=-1, type=int)
    parser.add_argument("-s", "--sample", type=float, default=1.0, help="Porcentaje de datos a usar (0.0 a 1.0)")

    args = parser.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args


def clean_text_bilingual(text):
    #Función de limpieza bilingüe con manejo de negaciones
    stemmers = {'es': SnowballStemmer('spanish'), 'en': SnowballStemmer('english')}
    stop_words = set(stopwords.words('spanish')).union(set(stopwords.words('english')))
    negaciones = {'no', 'not', 'neither', 'nor', 'none', 'never', 'nada', 'ni', 'nunca', 'tampoco'}
    stop_words = stop_words - negaciones

    text = str(text).lower()
    # Detección: si contiene palabras clave en español, usa stemmer ES
    spanish_markers = {'el', 'la', 'que', 'de', 'y', 'en'}
    lang = 'es' if len(spanish_markers.intersection(set(text.split()))) > 1 else 'en'
    stemmer = stemmers[lang]

    clean_text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    tokens = word_tokenize(clean_text, language='spanish' if lang == 'es' else 'english')
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned)


def balancear_con_ollama(df, target_col, text_col):
    counts = df[target_col].value_counts()
    max_size = counts.max()
    augmented_rows = []

    # 1. Definimos la "personalidad" y ejemplos para Ollama (Few-Shot)
    base_messages = [
        {
            "role": "system",
            "content": "You are a linguistic assistant. Your task is to paraphrase reviews briefly while keeping the same sentiment. Use the format: Original => Paraphrased"
        },
        {
            "role": "user",
            "content": "The app doesn't work well => The application works poorly"
        },
        {
            "role": "user",
            "content": "Me encanta esta interfaz => La interfaz me parece excelente"
        }
    ]

    for label, count in counts.items():
        if count < max_size:
            # Limitamos la generación al doble de la clase actual para evitar ruido
            n_to_generate = min(max_size - count, count)

            print(Fore.MAGENTA + f"[*] Generando {n_to_generate} variaciones para la clase: {label}" + Fore.RESET)

            # Seleccionamos las reseñas que vamos a parafrasear
            samples = df[df[target_col] == label].sample(n_to_generate, replace=True)

            for _, row in samples.iterrows():
                try:
                    # 2. Creamos el mensaje específico para esta fila
                    prompt = {"role": "user", "content": f"{row[text_col]} =>"}

                    # 2.5. Sacamos el máximo de palabras con cierta holgura
                    max = int(len(row[text_col].split()) +  2 * np.sqrt(len(row[text_col].split())))

                    # 3. Llamada a Ollama enviando el contexto completo
                    response = chat(
                        model="llama3:8b-text-q2_K",
                        messages=base_messages + [prompt],
                        options={"num_predict": max}
                    )

                    # 4. Nos quedamos solo con la primera linea
                    res = response.message.content.splitlines()[0]
                    # Crear la nueva fila
                    new_row = row.copy()
                    new_row[text_col] = res
                    augmented_rows.append(new_row)

                except Exception as e:
                    print(Fore.RED + f" Error en Ollama: {e}" + Fore.RESET)
                    continue

    # Unir datos originales con los sintéticos
    if augmented_rows:
        df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
        print(Fore.GREEN + f"[+] Aumento completado. Filas totales: {len(df_augmented)}" + Fore.RESET)
        return df_augmented

    return df


def preprocesar_entrenamiento(data, args):
    # 1. Mapeo de Score a Sentimiento
    mapeo = {1: 'negativo', 2: 'negativo', 3: 'neutral', 4: 'positivo', 5: 'positivo'}
    data[args.prediction] = data[args.prediction].map(mapeo)

    # 2. Identificación de columnas (solo review)
    text_cols = ['review']
    num_cols = []  # En este caso solo usamos review

    # SPLIT ANTES DE PROCESAR (Evita Data Leakage)
    df_train, df_dev = train_test_split(data, test_size=0.20, random_state=42, stratify=data[args.prediction])

    # 3. Sampling Generativo (Solo en entrenamiento)
    sampling = args.preprocessing.get("sampling", "none")
    if sampling == "generativo" or sampling == "over+gen" or sampling == "under+gen":
        df_train = balancear_con_ollama(df_train, args.prediction, 'review')

    # 4. Limpieza de Texto Bilingüe
    df_train['review'] = df_train['review'].apply(clean_text_bilingual)
    df_dev['review'] = df_dev['review'].apply(clean_text_bilingual)

    # 5. Vectorización (Fit solo en train)
    if args.preprocessing["text_process"] == "tf-idf":
        vec = TfidfVectorizer(max_features=5000,
                              ngram_range=(1,3),
                              min_df=2,
                              max_df=0.9,
                              sublinear_tf=True)
    else:
        vec = CountVectorizer(max_features=10000, ngram_range=(1, 2))

    xt_mat = vec.fit_transform(df_train['review'])
    xv_mat = vec.transform(df_dev['review'])

    # 6. Escalado (MaxAbsScaler no rompe la matriz sparse)
    scaler = MaxAbsScaler()
    xt = scaler.fit_transform(xt_mat)
    xv = scaler.transform(xv_mat)

    # 7. Codificación de variable objetivo
    le = LabelEncoder()
    yt = le.fit_transform(df_train[args.prediction])
    yd = le.transform(df_dev[args.prediction])

    # Retornamos los sets ya divididos para respetar la estructura de entrenamiento
    return xt, xv, yt, yd, vec, scaler, le, text_cols, num_cols, df_dev


def run_train(xt, xv, yt, yd, model, params, name, vec, scaler, le, text_cols, num_cols, args, df_dev):
    # Guardar set de validación
    df_dev.to_csv('output/traindev/dev_set.csv', index=False)

    # Balanceo clásico (Opcional si no se usa Ollama)
    sampling = args.preprocessing.get("sampling", "none")
    if sampling == "oversampling" or sampling == "over+gen":
        print(Fore.MAGENTA + "Aplicando oversampling..." + Fore.RESET)
        xt, yt = RandomOverSampler(random_state=42).fit_resample(xt, yt)
    elif sampling == "undersampling" or sampling == "under+gen":
        print(Fore.MAGENTA + "Aplicando undersampling..." + Fore.RESET)
        xt, yt = RandomUnderSampler(random_state=42).fit_resample(xt, yt)

    # Entrenamiento
    gs = GridSearchCV(model, params, cv=5, n_jobs=args.cpu, scoring=args.estimator)
    gs.fit(xt, yt)

    # Guardar artefacto
    pipeline = {
        'modelo': gs.best_estimator_,
        'vec': vec,
        'scaler': scaler,
        'le': le,
        'text_cols': text_cols,
        'num_cols': num_cols
    }

    with open(f'output/traindev/modelo_{name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print(Fore.CYAN + f"Modelo {name} | Mejor Score ({args.estimator}): {gs.best_score_:.4f}" + Fore.RESET)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists('output/traindev'): os.makedirs('output/traindev')
    nltk.download(['stopwords', 'punkt'], quiet=True)

    full_data = pd.read_csv(args.file, usecols=['review', args.prediction])
    if args.sample < 1.0:
        data = full_data.sample(frac=args.sample, random_state=42).reset_index(drop=True)
    else:
        data = full_data

    # Preprocesamiento
    xt, xv, yt, yd, vec, scaler, le, t_cols, n_cols, df_dev = preprocesar_entrenamiento(data, args)

    # Selección de algoritmo
    if args.algorithm == "kNN":
        p = args.kNN
        # Asegurar que n_neighbors sea una lista min, max step
        if isinstance(p['n_neighbors'], list) and len(p['n_neighbors']) == 3:
            p['n_neighbors'] = list(range(p['n_neighbors'][0], p['n_neighbors'][1] + 1, p['n_neighbors'][2]))
        run_train(xt, xv, yt, yd, KNeighborsClassifier(), p, "kNN", vec, scaler, le,
                  t_cols, n_cols, args, df_dev)

    elif args.algorithm == "decision_tree":
        run_train(xt, xv, yt, yd, DecisionTreeClassifier(), args.decision_tree, "decision_tree", vec, scaler, le,
                  t_cols, n_cols, args, df_dev)

    elif args.algorithm == "random_forest":
        run_train(xt, xv, yt, yd, RandomForestClassifier(), args.random_forest, "random_forest", vec, scaler, le,
                  t_cols, n_cols, args, df_dev)

    elif args.algorithm == "logistic_regression":
        run_train(xt, xv, yt, yd, LogisticRegression(max_iter=5000), args.logistic_regression, "logistic_regression",
                  vec, scaler, le, t_cols, n_cols, args, df_dev)

    elif args.algorithm == "naive_bayes":
        run_train(xt, xv, yt, yd, MultinomialNB(), args.naive_bayes,"naive_bayes",
                  vec, scaler, le, t_cols, n_cols, args, df_dev)

    else:
        print(Fore.RED + "Algoritmo no soportado." + Fore.RESET)
        sys.exit(-1)