# -*- coding: utf-8 -*-
"""
Autor: Marcos Cobo, Gabriel Gutiérrez, Aritz de la Pinta, Ibai Munné.
Script para la evaluación del dev
"""

import pandas as pd, pickle, os, argparse, re, string
from sklearn.metrics import classification_report, f1_score
from colorama import Fore, init
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Inicializar colorama para Windows
init(autoreset=True)


def clean_text_optimized(text):
    """
    IMPORTANTE: Esta función DEBE ser idéntica a la del script de train.
    Si usaste la versión con EXCLAM, usa esta.
    """
    st_es = SnowballStemmer('spanish')
    st_en = SnowballStemmer('english')
    stop_words = set(stopwords.words('spanish')).union(set(stopwords.words('english')))

    negaciones = {'no', 'ni', 'poco', 'tampoco', 'not', 'never', 'none', 'neither', 'without', 'sin'}
    stop_words = stop_words - negaciones

    text = str(text).lower()
    # Preservamos signos de exclamación como tokens (igual que en train)
    text = re.sub(r'(!+)', r' [EXCLAM] ', text)
    text = re.sub(r'(\?+)', r' [PREG] ', text)
    text = re.sub(r'[^a-zñáéíóú\s]', '', text)

    tokens = word_tokenize(text)
    is_es = len({'el', 'la', 'que', 'en'}.intersection(set(tokens))) > 0
    st = st_es if is_es else st_en

    cleaned = [st.stem(w) for w in tokens if w not in stop_words or w in negaciones]
    return ' '.join(cleaned)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-p", "--prediction", required=True, help="Columna objetivo (ej. score)")
    args = parse.parse_args()

    # Rutas
    dev_path = 'output/traindev/dev_set.csv'
    model_dir = 'output/traindev/'

    if not os.path.exists(dev_path):
        print(Fore.RED + f"Error: No existe {dev_path}.")
        exit()

    # Cargar datos de validación
    dev = pd.read_csv(dev_path)

    # Aseguramos que la columna de predicción tenga el mismo mapeo que en train
    mapeo = {1: 'negativo', 2: 'negativo', 3: 'neutral', 4: 'positivo', 5: 'positivo'}
    if dev[args.prediction].dtype in [int, float]:
        y_true_labels = dev[args.prediction].map(mapeo)
    else:
        y_true_labels = dev[args.prediction]

    model_files = [f for f in os.listdir(model_dir) if f.startswith('modelo_') and f.endswith('.pkl')]

    if not model_files:
        print(Fore.RED + "No se encontraron modelos en " + model_dir)
        exit()

    best_f1, best_model_name, best_report = -1, "", ""

    print(Fore.YELLOW + f"[*] Iniciando evaluación de {len(model_files)} modelos...\n")

    for f in model_files:
        try:
            with open(os.path.join(model_dir, f), 'rb') as file:
                p = pickle.load(file)

            # Limpieza de las reviews del set dev
            # Usamos 'review' por defecto ya que es la columna estándar del pipeline optimizado
            text_data = dev['review'].fillna('').apply(clean_text_optimized)

            # Transformación vectorial
            X_txt = p['vec'].transform(text_data)

            # Escalado (usando el scaler guardado)
            X_final = p['scaler'].transform(X_txt)

            # Predicción
            y_pred = p['modelo'].predict(X_final)

            # Encodeamos las etiquetas reales para comparar
            y_true_encoded = p['le'].transform(y_true_labels)

            score = f1_score(y_true_encoded, y_pred, average='macro')
            report = classification_report(
                y_true_encoded,
                y_pred,
                target_names=p['le'].classes_,
                zero_division=0
            )

            name = f.replace('modelo_', '').replace('.pkl', '')
            print(f"Modelo: {Fore.GREEN}{name:<20}{Fore.RESET} | F1-macro: {Fore.CYAN}{score:.4f}{Fore.RESET}")

            if score > best_f1:
                best_f1, best_model_name, best_report = score, name, report

        except Exception as e:
            print(Fore.RED + f"Error evaluando {f}: {e}")

    print("\n" + "=" * 60)
    print(f"{Fore.MAGENTA}GANADOR FINAL: {best_model_name}{Fore.RESET}")
    print(f"F1-macro en dev: {Fore.YELLOW}{best_f1:.4f}{Fore.RESET}")
    print("\nDetalle del Reporte:\n", best_report)
    print("=" * 60)