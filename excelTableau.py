import pandas as pd
import re

# El texto de tu informe LDA
texto_informe = """
Tema_0,"playlist (0.1931), play (0.1767), please (0.1303), need (0.0999), song (0.0998), better (0.0998), wo (0.0749), subscription (0.0386), service (0.0087), streaming (0.0082), update (0.0080), get (0.0078), even (0.0078), make (0.0077), time (0.0077), feel (0.0077), support (0.0077), issue (0.0077), also (0.0077)"
Tema_1,"support (0.2562), even (0.1053), also (0.0858), better (0.0687), issue (0.0685), subscription (0.0660), wo (0.0599), time (0.0542), get (0.0493), update (0.0474), song (0.0471), play (0.0328), feel (0.0262), service (0.0063), need (0.0054), streaming (0.0053), playlist (0.0053), make (0.0053), please (0.0053)"
Tema_2,"make (0.1704), feel (0.1163), update (0.1159), please (0.0895), even (0.0669), subscription (0.0615), issue (0.0614), playlist (0.0613), play (0.0401), get (0.0396), song (0.0347), time (0.0343), also (0.0341), better (0.0338), wo (0.0125), need (0.0071), service (0.0068), streaming (0.0068), support (0.0068)"
Tema_3,"streaming (0.2041), service (0.2020), get (0.1623), need (0.1391), time (0.1285), also (0.0469), subscription (0.0120), support (0.0114), even (0.0100), wo (0.0089), better (0.0087), make (0.0085), song (0.0085), update (0.0082), playlist (0.0082), feel (0.0082), play (0.0082), please (0.0082), issue (0.0082)"
"""


data = []

# Procesar cada línea del informe
lines = texto_informe.strip().split('\n')
for line in lines:
    # Extraer el nombre del tema (ej. Tema_0)
    tema_match = re.match(r'(Tema_\d+)', line)
    if tema_match:
        tema = tema_match.group(1)
        # Buscar todas las palabras y sus pesos: palabra (0.000)
        matches = re.findall(r'(\w+)\s+\(([\d.]+)\)', line)
        for palabra, peso in matches:
            data.append({
                'Aplicacion': 'TIDAL', # Puedes cambiarlo o hacerlo dinámico
                'Sentimiento': 'Todo',
                'Tema': tema,
                'Palabra': palabra,
                'Peso': float(peso)
            })

# Crear el DataFrame y exportar a Excel
df = pd.DataFrame(data)
df.to_excel('China.xlsx', index=False)

print("¡Archivo generado con éxito para Tableau!")