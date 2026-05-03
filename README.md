# SAD

Proyecto de la asignatura de Sistemas de Apoyo a la Decisión de la
carrera de Ingeniería Informática de la EHU.

------------------------------------------------------------------------

# Manual de Uso

## Requerimientos

-   Python 3.12
-   pip
-   conda // PyCharm
-   Para el uso del sampling generativo en el train, descargar Ollama si no se tiene
``` bash
ollama pull llama3:8b-text-q2_K
``` 

------------------------------------------------------------------------

## Instalación Conda

1.  Descargar `clustering.py`, `clasificador.json`, `train.py`, `dev.py`, `test.py` y
    `requirements.txt` en un directorio (Plantilla)
2.  Ejecuta:

``` bash
cd Plantilla
conda create -n proyecto-sad python=3.12
conda activate proyecto-sad
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Instalación PyCharm

1.  Importar `clustering.py`, `clasificador.json`, `train.py`, `dev.py`, `test.py` y
    `requirements.txt`
2.  Crear proyecto en PyCharm
3.  Instalar dependencias:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Ayuda

``` bash
python clustering.py --help
```

``` bash
python train.py --help
```

``` bash
python test.py --help
```

------------------------------------------------------------------------

## Uso

### Clustering

``` bash
python clustering.py -a lda -f data.csv -p score
```

### Train

``` bash
python train.py -a kNN -f data.csv -p score
```

### Dev
Evalua todos los modelos generados por el train en la carpeta output/traindev

``` bash
py .\dev.py -p score
```

### Test

``` bash
python test.py -a kNN -f data.csv -p score
```

------------------------------------------------------------------------

## JSON

```json
{
    "preprocessing": {
        "unique_category_threshold": 50,    // Numero de apariciones unicas para considerar una columna como categorica (int)
        "drop_features": [],                // Columnas a eliminar (lista de strings)
        "missing_values": "impute",          // Estrategia para tratar los valores nulos (impute, drop)
        "impute_strategy": "mean",           // Estrategia para imputar los valores nulos (mean, median, most\_frequent)
        "scaling": "standard",                  // Estrategia para escalar los valores (minmax, normalizer, maxabs, standard)
        "text_process": "tf-idf",            // Estrategia para procesar el texto (tf-idf, bow)
        "sampling": "none"                    // Estrategia para tratar el desbalanceo de clases (oversampling, undersampling, generativo, over+gen, under+gen)
    },
    "kNN": {
        "n_neighbors": [3, 13, 2],            // Numero de vecinos (min, max, step)
        "weights": ["uniform", "distance"],   // Peso de los vecinos (uniform, distance)
        "algorithm": ["auto"],                // Algoritmo para calcular los vecinos (auto, ball\_tree, kd\_tree, brute)
        "leaf_size": [30],                    // Tamaño de la hoja (lista de enteros)
        "p": [2]                              // Parametro de la distancia (1 para manhattan, 2 para euclidean)
    },
    "decision_tree": {
        "criterion": ["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10, 20, 30],         // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)
        "splitter": ["best"]                  // Estrategia para elegir la particion (best, random)
    },
    "random_forest": {
        "n_estimators": [50],                 // Numero de arboles (lista de enteros)
        "criterion": ["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10],                 // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)  
        "bootstrap": [false]                  // Si se deben usar muestras bootstrap (true, false)
    },
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],    //Parámetro de regularización inversa. Valores más bajos -> mayor regularización -> reduce sobreajuste
        "solver": ["saga"],                     //Algoritmo de optimización (lbfgs, liblinear, sag, saga)
        "class_weight": ["balanced"],           //Ajuste del peso de las clases (balanced)
        "max_iter": [5000]                      //Número máximo de iteraciones permitidas para la convergencia
    },
    "naive_bayes": {
        "alpha": [0.001, 0.01, 0.1, 1.0]        //Parámetro de suavizado de Laplace. Evita probabilidades cero
    },
    "k_means": {
        "n_clusters": [5,6,8],                  //Numero de clusters a probar
        "max_iter": [100],                      //Numero maximo de iteraccines
        "n_init": [10]                          //Cuantas veces se ejecutara el algoritmo completo con diferentes semillas aleatorias
    },
    "lda": {
        "n_clusters": [4,5,6],                  //Numero de clusters a probar
        "passes": 10                            //Cuantas veces el algoritmo recorrerá el conjunto de datos durante el entrenamiento
    }
}
```

---

# Documentación de Librerías

## 1. Gestión de Datos y Sistema
*   **pandas**: Utilizada para la carga, limpieza y manipulación de estructuras de datos tabulares (DataFrames). Es la base para el filtrado por metadatos como año y género.
*   **numpy**: Soporte para operaciones matemáticas y manejo de matrices numéricas, esencial para el procesamiento de vectores y etiquetas.
*   **argparse**: Permite la creación de una interfaz de línea de comandos para que los scripts acepten parámetros dinámicos (algoritmos, rutas de archivos, configuración de CPU).
*   **json**: Gestiona la persistencia de configuraciones externas a través del archivo `clasificador.json`.
*   **pickle**: Se emplea para serializar los modelos entrenados y los objetos de preprocesamiento, garantizando que el entorno de test sea idéntico al de entrenamiento.
*   **os / sys**: Utilizados para la gestión de rutas de archivos, creación de directorios de salida y control de ejecución del sistema.
*   **colorama**: Mejora la experiencia en la consola mediante la señalización visual de estados (éxitos, errores y advertencias) con colores.
*   **tqdm**: Implementa barras de progreso para monitorizar procesos iterativos largos, como la evaluación de múltiples clústeres.

---

## 2. Procesamiento de Lenguaje Natural (NLP)
*   **nltk (Natural Language Toolkit)**: Proporciona las herramientas necesarias para el tratamiento de texto:
    *   **Tokenización**: Segmentación de reseñas en palabras individuales.
    *   **Stopwords**: Eliminación de términos sin carga semántica en español e inglés.
    *   **Stemming / Lemmatization**: Normalización de palabras a sus raíces o lemas (SnowballStemmer y WordNetLemmatizer).
*   **re (Regular Expressions)**: Limpieza avanzada de cadenas mediante patrones para eliminar caracteres no alfabéticos y ruido textual.

---

## 3. Aprendizaje Automático y Preprocesamiento
*   **scikit-learn**: Librería central para la construcción de modelos:
    *   **Modelos**: Implementación de kNN, Decision Trees, Random Forest, Logistic Regression y Multinomial Naive Bayes.
    *   **Vectorización**: Conversión de texto a vectores numéricos mediante `TfidfVectorizer` y `CountVectorizer`.
    *   **Evaluación**: Uso de `GridSearchCV` para optimización de hiperparámetros y `f1_score` para validación.
    *   **Escalado**: `MaxAbsScaler` y `StandardScaler` para normalizar los datos de entrada sin destruir la estructura de matrices dispersas.
*   **imblearn**: Aplicación de técnicas de balanceo de datos mediante `RandomOverSampler` y `RandomUnderSampler` para corregir sesgos en las clases objetivo.

---

## 4. IA Generativa y Modelado Avanzado
*   **ollama**: Integración con modelos locales (Llama 3) para el aumento de datos generativo. Se utiliza para parafrasear reseñas reales y equilibrar el dataset de forma sintética manteniendo la semántica.
*   **gensim**: Especializada en el modelado de temas latentes. Implementa el algoritmo **LDA (Latent Dirichlet Allocation)** para extraer temáticas automáticas de las reseñas.
*   **pyLDAvis**: Genera visualizaciones interactivas de los temas detectados por LDA, permitiendo analizar la relevancia de los términos en cada grupo.
*   **matplotlib**: Utilizada para la generación de gráficos de diagnóstico, como las curvas de métricas de Silhouette y Coherence para determinar el número óptimo de grupos.

---