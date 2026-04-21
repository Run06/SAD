# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munne.
Script para la evaluación del set de desarrollo (dev).
"""

import pandas as pd, pickle, os, argparse
from sklearn.metrics import classification_report, f1_score
from colorama import Fore

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-p", "--prediction", required=True)
    args = parse.parse_args()

    if not os.path.exists('output/dev_set.csv'):
        print(
            Fore.RED + "Error: No existe 'output/dev_set.csv'. Debes ejecutar el script de entrenamiento primero." + Fore.RESET)
        exit()

    # Cargar datos de validación
    dev = pd.read_csv('output/dev_set.csv')
    y_true = dev[args.prediction]
    X_dev = dev.drop(columns=[args.prediction])

    model_files = [f for f in os.listdir('output') if f.endswith('.pkl')]

    if not model_files:
        print(Fore.RED + "No se encontraron modelos (.pkl) en la carpeta output." + Fore.RESET)
        exit()

    best_f1, best_model_name, best_report = -1, "", ""

    for f in model_files:
        with open(f'output/{f}', 'rb') as file:
            p = pickle.load(file)

        # Realizar predicción
        y_pred = p['modelo'].predict(X_dev)

        # Calcular Macro F1
        score = f1_score(y_true, y_pred, average='macro')

        # Obtener los nombres de las categorías (negative, neutral, positive)
        # p['le'] es el LabelEncoder que guardamos en el train
        target_names = p['le'].classes_

        # Generar el reporte detallado
        report = classification_report(y_true, y_pred, target_names=target_names)

        name = f.replace('modelo_', '').replace('.pkl', '')
        print(f"Evaluando modelo: {Fore.CYAN}{name}{Fore.RESET} | F1-Macro: {score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_model_name = name
            best_report = report

    # Mostrar resultados del Ganador con el formato solicitado
    print("\n" + "=" * 50)
    print(f"{Fore.GREEN}RESULTADOS DEL GANADOR: {best_model_name.upper()}{Fore.RESET}")
    print(f"Macro F1 en dev: {best_f1:.4f}")
    print(f"\nClassification report (dev):")
    print(best_report)
    print("=" * 50)