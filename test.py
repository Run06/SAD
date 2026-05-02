# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munne.
Script para la implementación del test.
"""

import sys, argparse, pandas as pd, pickle, os, json
from colorama import Fore
from train import clean_text_bilingual


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-a", "--algorithm", required=True)
    parser.add_argument("-p", "--prediction", required=True)

    args = parser.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    return args


if __name__ == "__main__":

    args = parse_args()

    path_modelo = f'output/traindev/modelo_{args.algorithm}.pkl'

    if not os.path.exists(path_modelo):
        print(Fore.RED + f"Modelo no encontrado: {path_modelo}" + Fore.RESET)
        sys.exit(1)

    with open(path_modelo, 'rb') as f:
        p = pickle.load(f)

    data = pd.read_csv(args.file)
    data_res = data.copy()

    os.makedirs('output/test', exist_ok=True)

    # limpieza texto (idéntica a train/dev)
    print(Fore.CYAN + "Limpiando texto..." + Fore.RESET)

    for col in p['text_cols']:
        data[col] = data[col].fillna('').apply(clean_text_bilingual)

    # vectorización
    txt_mat = p['vec'].transform(
        data[p['text_cols']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    )

    # escalado (sin fit)
    X_test = p['scaler'].transform(txt_mat)

    # predicción
    print(Fore.CYAN + "Prediciendo..." + Fore.RESET)

    preds = p['modelo'].predict(X_test)

    # inverse transform de etiquetas
    data_res[args.prediction] = p['le'].inverse_transform(preds)

    # guardar resultados
    output_path = 'output/test/data_prediction.csv'
    data_res.to_csv(output_path, index=False)

    print(Fore.GREEN + f"Guardado en {output_path}" + Fore.RESET)