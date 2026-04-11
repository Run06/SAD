

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
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
    over_under_sampling()

    drop_features()

    return data