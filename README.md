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

1.  Descargar `clasificador.json`, `train.py`, `dev.py`, `test.py` y
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

1.  Importar `clasificador.json`, `train.py`, `dev.py`, `test.py` y
    `requirements.txt`
2.  Crear proyecto en PyCharm
3.  Instalar dependencias:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Ayuda

``` bash
python train.py --help
```

``` bash
python test.py --help
```

------------------------------------------------------------------------

## Uso

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
        "unique\_category\_threshold": 50,    // Numero de apariciones unicas para considerar una columna como categorica (int)
        "drop\_features": [],                // Columnas a eliminar (lista de strings)
        "missing\_values": "impute",          // Estrategia para tratar los valores nulos (impute, drop)
        "impute\_strategy": "mean",           // Estrategia para imputar los valores nulos (mean, median, most\_frequent)
        "scaling": "standard",                  // Estrategia para escalar los valores (minmax, normalizer, maxabs, standard)
        "text\_process": "tf-idf",            // Estrategia para procesar el texto (tf-idf, bow)
        "sampling": "none"                    // Estrategia para tratar el desbalanceo de clases (oversampling, undersampling, generativo, over+gen, under+gen)
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
    },
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],    //Parámetro de regularización inversa. Valores más bajos -> mayor regularización -> reduce sobreajuste
        "solver": ["saga"],                     //Algoritmo de optimización (lbfgs, liblinear, sag, saga)
        "class_weight": ["balanced"],           //Ajuste del peso de las clases (balanced)
        "max_iter": [5000]                      //Número máximo de iteraciones permitidas para la convergencia
    },
    "naive_bayes": {
        "alpha": [0.001, 0.01, 0.1, 1.0]        //Parámetro de suavizado de Laplace. Evita probabilidades cero
    }
}
```