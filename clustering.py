# -*- coding: utf-8 -*-
import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import json
import csv
import os
from colorama import Fore
import matplotlib.pyplot as plt

# Sklearn (Para Clustering y Preprocesamiento)
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

vectorizer_obj = None

def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    parse = argparse.ArgumentParser(description="Algoritmo de Clustering y Topic Modeling")
    parse.add_argument("-f", "--file", help="Fichero csv", required=True)
    parse.add_argument("-a", "--algorithm", help="k_means o lda", required=True)
    parse.add_argument("-p", "--prediction", help="Columna para separar Positivas y Negativas", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="CPUs", required=False, default=-1, type=int)
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")
    parse.add_argument("-y", "--year", help="Filtrar por año (ej. 2016)", required=False, default=None)
    parse.add_argument("-g", "--gender", help="Filtrar por genero (ej. female o male)", required=False, default=None)
    parse.add_argument("-l", "--location", help="Filtrar por pais (ej. Finland)", required=False, default=None)

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
        print(Fore.RED + f"Error al cargar: {e}" + Fore.RESET)
        sys.exit(1)

def preprocesar_datos():
    global data, vectorizer_obj

    text_cols = ['review'] if 'review' in data.columns else []
    if not text_cols:
        print(Fore.RED + "Error: No se ha encontrado la columna 'review' en el CSV." + Fore.RESET)
        sys.exit(1)

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if args.prediction and args.prediction in numerical_cols:
        numerical_cols.remove(args.prediction)

    categorical_all = data.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'review' in categorical_all:
        categorical_all.remove('review')
    if args.prediction and args.prediction in categorical_all:
        categorical_all.remove(args.prediction)

    cat_cols = [c for c in categorical_all if data[c].nunique() <= args.preprocessing.get("unique_category_threshold", 51)]

    # Eliminar palabras que no aportan informacion
    stop_words_eng = set(stopwords.words('english'))
    stop_words_spa = set(stopwords.words('spanish'))
    stop_words = stop_words_eng.union(stop_words_spa)

    # Palabras personalizadas que sepamos que no aportan nada
    palabras_basura_dominio = {
        'app', 'music', 'tidal', 'spotify', 'sound', 'qualiti', 'quality',
        'song', 'good', 'great', 'best', 'love', 'work', 'use', 'like', 'far', 'ever',
        'aplicacion', 'musica', 'bueno', 'buena', 'mejor'
    }
    stop_words = stop_words.union(palabras_basura_dominio)

    lemmatizer = WordNetLemmatizer()
    data['review'] = data['review'].fillna("")

    print("- Limpiando y lematizando textos...")
    data['tokens_clean'] = data['review'].astype(str).apply(lambda x: [
        lemmatizer.lemmatize(w) for w in word_tokenize(x.lower())
        if w not in stop_words and w not in string.punctuation and w.isalpha()
    ])

    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    strat = args.preprocessing.get("missing_values", "impute")
    if strat == "drop":
        data = data.dropna(subset=numerical_cols + cat_cols + ['tokens_clean']).reset_index(drop=True)
    elif strat == "impute":
        imp = args.preprocessing.get("impute_strategy", "mean")
        for col in numerical_cols:
            val = data[col].mean() if imp == "mean" else data[col].median()
            data[col] = data[col].fillna(val)
        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Missing")

    if numerical_cols:
        sc_type = args.preprocessing.get("scaling", "standard")
        scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "maxabs": MaxAbsScaler(), "normalizer": Normalizer()}
        if sc_type in scalers:
            data[numerical_cols] = scalers[sc_type].fit_transform(data[numerical_cols])

    # TF-IDF para el k_means
    method = args.preprocessing.get("text_process", "tf-idf")
    vec = TfidfVectorizer(max_df=0.85, min_df=0.01) if method == "tf-idf" else CountVectorizer(max_df=0.85, min_df=0.01)

    combined_text = data['tokens_clean'].apply(lambda x: ' '.join(x))
    matrix = vec.fit_transform(combined_text)
    vectorizer_obj = vec

    nombres_palabras = ["word_" + w for w in vec.get_feature_names_out()]
    text_df = pd.DataFrame(matrix.toarray(), columns=nombres_palabras)

    data.reset_index(drop=True, inplace=True)
    data = data.drop(columns=['review'])
    data = pd.concat([data, text_df], axis=1)

    if args.preprocessing.get("drop_features", []):
        data.drop(columns=args.preprocessing["drop_features"], inplace=True, errors='ignore')


def run_lda(params, name, subset_data):
    textos = subset_data['tokens_clean'].tolist()

    if not textos or len(textos) == 0:
        print(Fore.RED + f"No hay palabras suficientes para analizar en: {name}" + Fore.RESET)
        return

    print(Fore.YELLOW + f"Preparando Diccionario y Corpus para LDA ({name})..." + Fore.RESET)

    dictionary = corpora.Dictionary(textos)
    dictionary.filter_extremes(no_below=5, no_above=0.85)

    corpus = [dictionary.doc2bow(text) for text in textos]

    n_clusters_list = params.get("n_clusters", [2, 3, 4, 5])
    pasadas = params.get("passes", 10)

    best_score = -1
    best_model = None
    best_k = -1
    results = []

    print(Fore.YELLOW + f"Evaluando LDA para K en {n_clusters_list} ({name})..." + Fore.RESET)

    for k in tqdm(n_clusters_list, desc='Evaluando Temas LDA'):
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=pasadas,
            alpha='auto',
            per_word_topics=True
        )

        coherence_model_lda = CoherenceModel(model=lda_model, texts=textos, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()

        results.append({'k': k, 'coherence': coherence_score})

        # Nos quedamos con el modelo que tenga la nota más alta
        if coherence_score > best_score:
            best_score = coherence_score
            best_model = lda_model
            best_k = k

    plt.figure(figsize=(8, 5))
    plt.plot([r['k'] for r in results], [r['coherence'] for r in results], marker='o', linestyle='-', color='purple',
             linewidth=2)
    plt.title(f'Calidad de Temas LDA (Coherence Score) - {name.upper()}')
    plt.xlabel('Número de Temas (K)')
    plt.ylabel('Coherence Score (Más alto es mejor)')
    plt.grid(True, linestyle='--', alpha=0.7)

    grafico_path = f'output/grafico_coherence_lda_{name}.png'
    plt.savefig(grafico_path, bbox_inches='tight')
    plt.close()
    print(Fore.CYAN + f"Gráfico de Coherencia (LDA) guardado en: {grafico_path}" + Fore.RESET)

    palabras_por_cluster = {}
    for idx, topic in best_model.show_topics(formatted=False, num_topics=best_k, num_words=30):
        top_words = [f"{word} ({prop:.4f})" for word, prop in topic]
        palabras_por_cluster[f"Tema_{idx}"] = top_words

    save_clustering_report(best_model, f"lda_{name}", results, best_score, {'num_topics': best_k}, palabras_por_cluster)

    print(Fore.YELLOW + "Generando visualización interactiva del mejor modelo..." + Fore.RESET)
    try:
        vis = gensimvis.prepare(best_model, corpus, dictionary, mds='mmds')
        html_path = f'output/dashboard_lda_{name}.html'
        pyLDAvis.save_html(vis, html_path)
        print(Fore.GREEN + f"Dashboard interactivo guardado en: {html_path}" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"Error al generar la visualización: {e}" + Fore.RESET)


def run_clustering(params, name, subset_data):
    x = subset_data.copy()
    if args.prediction and args.prediction in x.columns:
        x = x.drop(columns=[args.prediction])
    if 'tokens_clean' in x.columns:
        x = x.drop(columns=['tokens_clean'])

    text_cols_tf = ["word_" + w for w in vectorizer_obj.get_feature_names_out()] if vectorizer_obj else []
    x_clustering = x[text_cols_tf] if text_cols_tf else x

    if x_clustering.empty or len(x_clustering.columns) == 0:
        return

    n_clusters_list = params.get("n_clusters", [2, 3, 4, 5])
    max_iter = params.get("max_iter", [100])[0]
    n_init = params.get("n_init", [10])[0]

    best_score, best_model, best_params, best_labels = -1, None, None, None
    results = []

    print(Fore.YELLOW + f"Evaluando K-Means para K en {n_clusters_list} ({name})..." + Fore.RESET)
    for k in tqdm(n_clusters_list, desc='Evaluando Clusters'):
        km = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=42)
        labels = km.fit_predict(x_clustering)
        if 1 < k < len(x_clustering):
            sil_score = silhouette_score(x_clustering, labels)
            db_score = davies_bouldin_score(x_clustering, labels)
        else:
            sil_score, db_score = -1, -1
        results.append({'k': k, 'silhouette': sil_score, 'davies_bouldin': db_score})
        if sil_score > best_score:
            best_score, best_model, best_params, best_labels = sil_score, km, {'n_clusters': k, 'max_iter': max_iter,
                                                                               'n_init': n_init}, labels

    plt.figure(figsize=(8, 5))
    plt.plot([r['k'] for r in results], [r['silhouette'] for r in results], marker='o', linestyle='-', color='blue',
             linewidth=2)
    plt.title(f'Calidad de Clusters (Silhouette Score) - {name.upper()}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score (Más cerca de 1 es mejor)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'output/grafico_silhouette_{name}.png', bbox_inches='tight')
    plt.close()

    palabras_por_cluster = {}
    features = x_clustering.columns
    for i in range(best_params['n_clusters']):
        indices_ordenados = best_model.cluster_centers_[i].argsort()[::-1]
        top_words = []
        for idx in indices_ordenados[:30]:
            word = features[idx].replace("word_", "")
            peso = best_model.cluster_centers_[i][idx]
            top_words.append(f"{word} ({peso:.4f})")

        palabras_por_cluster[f"Cluster_{i}"] = top_words

    save_clustering_report(best_model, name, results, best_score, best_params, palabras_por_cluster)

    try:
        datos_finales = x.copy()
        datos_finales['Cluster_Asignado'] = best_labels
        datos_finales.to_csv(f'output/datos_con_clusters_{name}.csv', index=False)
    except Exception as e:
        print(Fore.RED + f"Error al guardar los datos agrupados: {e}" + Fore.RESET)


def save_clustering_report(model, name, results, best_score, best_params, palabras_dict):
    try:
        with open(f'output/modelo_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(f'output/resultado_{name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'--- INFORME: {name.upper()} ---'])
            writer.writerow(['Metrica', 'Valor'])
            writer.writerow(['Calidad / Puntuacion', best_score])
            writer.writerow(['Numero optimo de grupos', best_params.get('n_clusters', best_params.get('num_topics'))])
            writer.writerow(['--- RAZONES DETECTADAS POR GRUPO ---'])
            for cluster, words in palabras_dict.items():
                writer.writerow([cluster, ", ".join(words)])

    except Exception as e:
        print(Fore.RED + f"Error al guardar reporte: {e}" + Fore.RESET)


if __name__ == "__main__":
    np.random.seed(42)
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    if not os.path.exists('output'): os.makedirs('output')

    print("- Cargando datos...")
    data = load_data(args.file)


    #FILTRADO OPCIONAL POR AÑO Y GÉNERO (antes de procesar)
    if args.year:
        col_date = 'Date' if 'Date' in data.columns else 'date' if 'date' in data.columns else None
        if col_date:
            print(Fore.CYAN + f"- Aplicando filtro de Año: {args.year}" + Fore.RESET)
            data = data[data[col_date].astype(str).str.contains(str(args.year), na=False)]
        else:
            print(Fore.YELLOW + "- Advertencia: No se encontró la columna de fecha para filtrar." + Fore.RESET)

    if args.gender:
        col_gender = 'Gender' if 'Gender' in data.columns else 'gender' if 'gender' in data.columns else None
        if col_gender:
            print(Fore.CYAN + f"- Aplicando filtro de Género: {args.gender}" + Fore.RESET)
            data = data[data[col_gender].astype(str).str.lower() == str(args.gender).lower()]
        else:
            print(Fore.YELLOW + "- Advertencia: No se encontró la columna de género para filtrar." + Fore.RESET)

    if args.location:
        col_loc = 'Location' if 'Location' in data.columns else 'location' if 'location' in data.columns else None
        if col_loc:
            print(Fore.CYAN + f"- Aplicando filtro de País: {args.location}" + Fore.RESET)
            # Usamos str.contains para que busque "Finland" dentro de "Helsinki, Finland"
            data = data[data[col_loc].astype(str).str.contains(str(args.location), case=False, na=False)]
        else:
            print(Fore.YELLOW + "- Advertencia: No se encontró la columna de location para filtrar." + Fore.RESET)

    if data.empty:
        print(Fore.RED + "Error: Los filtros dejaron el dataset vacío. Revisa el año o género indicado." + Fore.RESET)
        sys.exit(1)

        # Crear el sufijo para que los archivos no se sobrescriban
    sufijo = ""
    if args.year: sufijo += f"_{args.year}"
    if args.gender: sufijo += f"_{args.gender}"
    #FIN FILTRADO


    print("- Descargando diccionarios...")
    nltk.download(['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4'], quiet=True)

    print("- Preprocesando datos...")
    score_col = args.prediction if args.prediction in data.columns else None
    if score_col:
        scores_originales = data[score_col].copy()

    preprocesar_datos()

    if args.debug:
        data.to_csv('output/data-processed.csv', index=False)
        print(Fore.GREEN + "CSV de debug guardado." + Fore.RESET)

    print("- Ejecutando algoritmo...")

    if args.algorithm in ["k_means", "lda"]:
        if score_col:
            print(
                Fore.CYAN + f"\nSeparando datos en Positivos/Negativos usando la columna '{score_col}'..." + Fore.RESET)
            data[score_col] = scores_originales
            mediana = data[score_col].median()
            df_positivas = data[data[score_col] > mediana].copy()
            df_negativas = data[data[score_col] <= mediana].copy()

            print(Fore.GREEN + f"\n[=== INICIANDO {args.algorithm.upper()}: RESEÑAS POSITIVAS ===]" + Fore.RESET)
            nombre_pos = f"positivas{sufijo}"
            if args.algorithm == "k_means": run_clustering(args.k_means, nombre_pos, subset_data=df_positivas)
            if args.algorithm == "lda": run_lda(args.lda, nombre_pos, subset_data=df_positivas)

            print(Fore.RED + f"\n[=== INICIANDO {args.algorithm.upper()}: RESEÑAS NEGATIVAS ===]" + Fore.RESET)
            nombre_neg = f"negativas{sufijo}"
            if args.algorithm == "k_means": run_clustering(args.k_means, nombre_neg, subset_data=df_negativas)
            if args.algorithm == "lda": run_lda(args.lda, nombre_neg, subset_data=df_negativas)
        else:
            print(Fore.YELLOW + f"\nEjecutando {args.algorithm.upper()} general..." + Fore.RESET)
            nombre_gen = f"general{sufijo}"
            if args.algorithm == "k_means": run_clustering(args.k_means, nombre_gen, subset_data=data)
            if args.algorithm == "lda": run_lda(args.lda, nombre_gen, subset_data=data)
    else:
        print(Fore.RED + "Algoritmo no soportado. Usa k_means o lda." + Fore.RESET)