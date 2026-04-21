# SAD
Proyecto de la asignatura de Sistemas de Apoyo a la Decisión de la carrera de Ingeniería Informática de la EHU.

# Manual de Uso

## Requerimientos

* Python 3.12
* pip
* conda // PyCharm

## Instalación Conda

1. Descargar clasificador.json, test.py, trainDev.py y requirements.txt en un directorio Plantilla
2. Ejecuta:
	cd Plantilla
	conda create -n proyecto-plantilla python=3.12
	conda activate proyecto-plantilla
	pip install -r requirements.txt

		(los nombres del directorio y entorno de conda pueden cambiar)

## Instalación PyCharm

1. Instalar archivos clasificador.json, test.py, trainDev.py y requirements.txt
2. Una vez dentro de PyCharm. Importar los archivos en un nuevo proyecto.
3. Instalar las dependencias con pip:

   pip install -r requirements.txt



## Ayuda

```bash
python train.py --help
=== trainDev ===
usage: train.py \[-h] -f FILE -a ALGORITHM -p PREDICTION \[-e ESTIMATOR] \[-c CPU] \[-v] \[--debug]

Codigo para entrenar un modelo a partir de un csv.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Fichero csv (/Path\_to\_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a ejecutar (kNN, decision\_tree o random\_forest)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -e ESTIMATOR, --estimator ESTIMATOR
                        Estimador a utilizar para elegir el mejor modelo (ver más abajo diferentes estimadores)
  -c CPU, --cpu CPU     Número de CPUs a utilizar \[-1 para usar todos]
  -v, --verbose         Muestra las metricas por la termina
  --debug               Modo debug \[Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]
```

```bash
python test.py --help
=== test ===
usage: test.py \[-h] -f FILE -a ALGORITHM -p PREDICTION  \[-c CPU] \[--debug]

Codigo para predecir instancias en un csv.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Fichero csv (/Path\_to\_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a ejecutar (kNN, decision\_tree o random\_forest)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -c CPU, --cpu CPU     Número de CPUs a utilizar \[-1 para usar todos]
  --debug               Modo debug \[Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]
```



## Uso

Basico (ejemplo train)

```bash
python train.py -a kNN -f iris.csv -p Especie
```

Avanzado (ejemplo train)

```bash
python train.py -a kNN -f iris.csv -p Especie -e accuracy -c 4 -v --debug
```



## JSON

```json
{
    "preprocessing": {
        "unique\_category\_threshold": 50,      // Numero de apariciones unicas para considerar una columna como categorica (int)
        "drop\_features": \[],                  // Columnas a eliminar (lista de strings)
        "missing\_values": "impute",           // Estrategia para tratar los valores nulos (impute, drop)
        "impute\_strategy": "mean",            // Estrategia para imputar los valores nulos (mean, median, most\_frequent)
        "scaling": "minmax",                  // Estrategia para escalar los valores (minmax, normalizer, maxabs, standard)
        "text\_process": "tf-idf",             // Estrategia para procesar el texto (tf-idf, bow)
        "sampling": "oversampling"            // Estrategia para tratar el desbalanceo de clases (oversampling, undersampling)
    },
    "kNN": {
        "n\_neighbors": \[3, 13, 2],            // Numero de vecinos (min, max, step)
        "weights": \["uniform", "distance"],   // Peso de los vecinos (uniform, distance)
        "algorithm": \["auto"],                // Algoritmo para calcular los vecinos (auto, ball\_tree, kd\_tree, brute)
        "leaf\_size": \[30],                    // Tamaño de la hoja (lista de enteros)
        "p": \[2]                              // Parametro de la distancia (1 para manhattan, 2 para euclidean)
    },
    "decision\_tree": {
        "criterion": \["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max\_depth": \[5, 10, 20, 30],         // Profundidad maxima del arbol (lista de enteros)
        "min\_samples\_split": \[2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min\_samples\_leaf": \[1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max\_features": \["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)
        "splitter": \["best"]                  // Estrategia para elegir la particion (best, random)
    },
    "random\_forest": {
        "n\_estimators": \[50],                 // Numero de arboles (lista de enteros)
        "criterion": \["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max\_depth": \[5, 10],                 // Profundidad maxima del arbol (lista de enteros)
        "min\_samples\_split": \[2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min\_samples\_leaf": \[1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max\_features": \["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)  
        "bootstrap": \[false]                  // Si se deben usar muestras bootstrap (true, false)
    }
}
```

## ESTIMATOR
```estimator
Scoring string name             Function                                Comment
'accuracy'                      metrics.accuracy_score
‘balanced_accuracy’             metrics.balanced_accuracy_score
‘average_precision’             metrics.average_precision_score
‘f1’                            metrics.f1_score                        for binary targets
‘f1_micro’                      metrics.f1_score                        micro-averaged
‘f1_macro’                      metrics.f1_score                        macro-averaged
‘precision’                     metrics.precision_score
‘recall’                        metrics.recall_score
```

