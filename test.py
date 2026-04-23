# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munne.
Script para la implementación del test.
"""

import sys, argparse, pandas as pd, numpy as np, string, pickle, json, os, nltk
from colorama import Fore
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Descarga de recursos necesarios
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("-f", "--file", required=True)
    parse.add_argument("-a", "--algorithm", required=True)
    parse.add_argument("-p", "--prediction", required=True)
    args = parse.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items(): setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = parse_args()
    path_modelo = f'output/traindev/modelo_{args.algorithm}.pkl'

    if not os.path.exists(path_modelo):
        print(Fore.RED + f"Error: No existe el modelo {path_modelo}" + Fore.RESET)
        sys.exit(1)

    with open(path_modelo, 'rb') as f:
        p = pickle.load(f)

    data = pd.read_csv(args.file)
    data_res = data.copy()

    if not os.path.exists('output/test'):
        os.makedirs('output/test')

    # Preprocesado de texto
    lang = args.preprocessing.get("language", "english")
    stemmer = SnowballStemmer(lang)
    stop = set(stopwords.words(lang))

    print(Fore.CYAN + "Procesando columnas de texto..." + Fore.RESET)
    for col in p['text_cols']:
        data[col] = data[col].astype(str).apply(lambda x: ' '.join(
            [stemmer.stem(w) for w in word_tokenize(x.lower())
             if w not in stop and w not in string.punctuation]))

    # Vectorización
    txt_mat = p['vec'].transform(data[p['text_cols']].apply(lambda x: ' '.join(x.astype(str)), axis=1))
    df_txt = pd.DataFrame(txt_mat.toarray(), columns=p['vec'].get_feature_names_out())

    # Manejo de columnas numéricas
    # Se verifica si num_cols existe y si contiene elementos
    if 'num_cols' in p and len(p['num_cols']) > 0:
        print(Fore.CYAN + "Procesando columnas numéricas..." + Fore.RESET)
        X_num = data[p['num_cols']].fillna(0)

        # Verificación de si el scaler está entrenado
        if hasattr(p['scaler'], 'mean_'):
            X_num_scaled = p['scaler'].transform(X_num)
        else:
            print(Fore.YELLOW + "Aviso: Scaler no entrenado. Ajustando con datos actuales." + Fore.RESET)
            X_num_scaled = p['scaler'].fit_transform(X_num)

        df_num = pd.DataFrame(X_num_scaled, columns=p['num_cols'])
        X_test = pd.concat([df_num.reset_index(drop=True), df_txt.reset_index(drop=True)], axis=1)
    else:
        print(Fore.YELLOW + "No hay columnas numéricas definidas. Usando solo texto." + Fore.RESET)
        X_test = df_txt.reset_index(drop=True)

    # Predicción
    print(Fore.CYAN + "Realizando predicciones..." + Fore.RESET)
    preds = p['modelo'].predict(X_test)
    data_res[args.prediction] = p['le'].inverse_transform(preds)

    # Guardado de resultados
    data_res.to_csv('output/test/data_prediction.csv', index=False)
    print(Fore.GREEN + "Resultados guardados en output/test/data_prediction.csv" + Fore.RESET)