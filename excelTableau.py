import pandas as pd
import re

# El texto de tu informe LDA
texto_informe = """
Tema_0,"better (0.0715), artist (0.0494), awesome (0.0296), much (0.0261), pay (0.0193), way (0.0150), selection (0.0147), get (0.0126), streaming (0.0121), apps (0.0111), listen (0.0110), audio (0.0106), well (0.0104), sound (0.0098), free (0.0097), perfect (0.0096), apple (0.0093), ui (0.0090), new (0.0087), favorite (0.0069), top (0.0069), exceptional (0.0064), premium (0.0063), super (0.0062), frustrating (0.0061), actually (0.0059), recommend (0.0059), hifi (0.0058), guy (0.0055), pretty (0.0054)"
Tema_1,"excellent (0.0222), nice (0.0164), update (0.0150), issue (0.0117), hifi (0.0107), phone (0.0105), work (0.0104), high (0.0095), audio (0.0094), time (0.0088), using (0.0079), really (0.0077), new (0.0073), thing (0.0073), keep (0.0073), play (0.0070), device (0.0067), would (0.0064), well (0.0064), used (0.0064), back (0.0061), also (0.0060), year (0.0059), system (0.0059), price (0.0058), home (0.0054), support (0.0053), android (0.0053), headphone (0.0053), service (0.0052)"
Tema_2,"amazing (0.0260), song (0.0248), experience (0.0203), make (0.0182), playlist (0.0158), really (0.0153), one (0.0140), could (0.0133), listen (0.0122), feature (0.0113), many (0.0106), using (0.0099), want (0.0097), please (0.0094), would (0.0094), get (0.0088), day (0.0086), every (0.0085), absolutely (0.0084), album (0.0082), add (0.0081), listening (0.0078), feel (0.0077), know (0.0075), definitely (0.0074), subscription (0.0068), less (0.0067), new (0.0066), improved (0.0065), year (0.0064)"
Tema_3,"streaming (0.0533), service (0.0436), audio (0.0272), easy (0.0200), interface (0.0166), user (0.0156), always (0.0132), search (0.0128), find (0.0121), need (0.0120), platform (0.0119), artist (0.0089), experience (0.0082), thanks (0.0080), playlist (0.0079), also (0.0078), lot (0.0065), high (0.0063), make (0.0059), way (0.0058), option (0.0056), everything (0.0055), one (0.0055), master (0.0055), superior (0.0054), content (0.0054), looking (0.0053), simply (0.0050), appealing (0.0050), hifi (0.0049)"
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
                'Sentimiento': 'Positivo',
                'Tema': tema,
                'Palabra': palabra,
                'Peso': float(peso)
            })

# Crear el DataFrame y exportar a Excel
df = pd.DataFrame(data)
df.to_excel('LDA_Tidal_Estructurado.xlsx', index=False)

print("¡Archivo generado con éxito para Tableau!")