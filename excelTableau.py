import pandas as pd
import re

# El texto de tu informe LDA
texto_informe = """
Tema_0,"ca (0.0309), month (0.0191), phone (0.0166), end (0.0163), free (0.0158), bad (0.0152), keep (0.0152), listen (0.0143), able (0.0133), day (0.0129), pc (0.0126), try (0.0123), account (0.0115), problem (0.0113), trial (0.0112), connect (0.0110), less (0.0110), control (0.0109), even (0.0104), charged (0.0104), without (0.0102), make (0.0101), paying (0.0101), subscription (0.0100), canceled (0.0100), useless (0.0091), apps (0.0090), mode (0.0088), find (0.0087), offline (0.0086)"
Tema_1,"even (0.0280), play (0.0261), listen (0.0259), pay (0.0200), wo (0.0197), playing (0.0189), one (0.0183), want (0.0180), playlist (0.0178), album (0.0175), song (0.0147), artist (0.0144), let (0.0132), free (0.0130), ad (0.0127), without (0.0112), subscription (0.0112), track (0.0100), ca (0.0099), able (0.0097), support (0.0093), update (0.0090), make (0.0089), middle (0.0088), random (0.0084), get (0.0083), paying (0.0078), high (0.0078), need (0.0077), actually (0.0075)"
Tema_2,"issue (0.0343), time (0.0270), song (0.0248), play (0.0187), service (0.0179), downloaded (0.0135), support (0.0131), subscription (0.0125), stop (0.0124), connection (0.0117), ca (0.0114), fix (0.0113), every (0.0098), change (0.0097), feel (0.0096), year (0.0094), offline (0.0089), crash (0.0088), phone (0.0086), customer (0.0085), get (0.0083), start (0.0082), say (0.0079), waste (0.0079), useless (0.0078), work (0.0077), back (0.0077), something (0.0074), connect (0.0073), playback (0.0073)"
Tema_3,"even (0.0249), ca (0.0227), keep (0.0202), error (0.0195), account (0.0185), service (0.0168), get (0.0156), time (0.0150), log (0.0150), play (0.0149), still (0.0129), android (0.0127), better (0.0127), free (0.0126), using (0.0125), login (0.0120), plan (0.0114), playlist (0.0113), tried (0.0105), back (0.0103), terrible (0.0101), suck (0.0101), going (0.0094), card (0.0086), option (0.0086), subscription (0.0086), password (0.0084), streaming (0.0083), need (0.0078), crashing (0.0076)"
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
                'Sentimiento': 'Negativo',
                'Tema': tema,
                'Palabra': palabra,
                'Peso': float(peso)
            })

# Crear el DataFrame y exportar a Excel
df = pd.DataFrame(data)
df.to_excel('Neg_2016_Male.xlsx', index=False)

print("¡Archivo generado con éxito para Tableau!")