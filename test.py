# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munne.
Script para la implementación del test.
"""

import sys, argparse, pd, numpy as np, string, pickle, json, os, nltk
from colorama import Fore
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("-f", "--file", required=True)
    parse.add_argument("-a", "--algorithm", required=True)
    parse.add_argument("-p", "--prediction", required=True)
    args = parse.parse_args()
    with open('clasificador.json') as f: config = json.load(f)
    for k, v in config.items(): setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(f'output/modelo_{args.algorithm}.pkl', 'rb') as f:
        p = pickle.load(f)

    data = pd.read_csv(args.file)
    data_res = data.copy()

    # Preprocesado usando los objetos del entrenamiento (p['vec'], p['scaler'])
    lang = args.preprocessing.get("language", "english")
    stemmer, stop = SnowballStemmer(lang), set(stopwords.words(lang))

    for col in p['text_cols']:
        data[col] = data[col].astype(str).apply(lambda x: ' '.join(sorted(
            [stemmer.stem(w) for w in word_tokenize(x.lower()) if w not in stop and w not in string.punctuation])))

    txt_mat = p['vec'].transform(data[p['text_cols']].apply(lambda x: ' '.join(x.astype(str)), axis=1))
    data[p['num_cols']] = p['scaler'].transform(data[p['num_cols']].fillna(0))

    X_test = pd.concat([data[p['num_cols']].reset_index(drop=True),
                        pd.DataFrame(txt_mat.toarray(), columns=p['vec'].get_feature_names_out())], axis=1)

    # Predicción y des-codificación (0,1,2 -> baja, media, alta)
    preds = p['modelo'].predict(X_test)
    data_res[args.prediction] = p['le'].inverse_transform(preds)

    data_res.to_csv('output/data-prediction.csv', index=False)
    print(Fore.GREEN + "Predicciones guardadas en output/data-prediction.csv" + Fore.RESET)
