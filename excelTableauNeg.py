import pandas as pd
import re

# El texto de tu informe LDA de NEGATIVAS
texto_negativas = """
Tema_0,"playlist (0.0369), song (0.0261), artist (0.0175), track (0.0153), album (0.0152), play (0.0120), also (0.0111), add (0.0106), would (0.0103), one (0.0099), please (0.0098), want (0.0084), option (0.0081), feature (0.0080), ca (0.0075), phone (0.0074), search (0.0072), android (0.0071), make (0.0070), really (0.0066), new (0.0063), shuffle (0.0060), listen (0.0059), back (0.0058), way (0.0058), need (0.0057), thing (0.0056), go (0.0054), find (0.0054), update (0.0053)"
Tema_1,"make (0.0311), experience (0.0271), user (0.0239), interface (0.0236), less (0.0183), feel (0.0182), using (0.0165), android (0.0162), audio (0.0162), many (0.0145), watch (0.0141), playback (0.0141), auto (0.0135), saved (0.0120), thing (0.0103), issue (0.0103), get (0.0100), slow (0.0097), friendly (0.0092), recommendation (0.0089), samsung (0.0087), problem (0.0086), always (0.0086), frustrating (0.0086), know (0.0085), people (0.0084), galaxy (0.0079), occasionally (0.0074), generally (0.0072), one (0.0071)"
Tema_2,"better (0.0272), audio (0.0186), service (0.0162), would (0.0136), streaming (0.0136), need (0.0124), could (0.0124), artist (0.0108), feature (0.0093), google (0.0091), much (0.0089), really (0.0089), search (0.0082), also (0.0080), high (0.0076), find (0.0073), bit (0.0071), hifi (0.0068), star (0.0067), still (0.0066), integration (0.0065), available (0.0063), ui (0.0062), master (0.0061), wish (0.0061), nice (0.0058), pretty (0.0057), lot (0.0057), overall (0.0056), selection (0.0056)"
Tema_3,"play (0.0192), issue (0.0178), time (0.0177), downloaded (0.0148), playing (0.0138), stop (0.0134), offline (0.0133), keep (0.0123), fix (0.0118), phone (0.0117), song (0.0115), even (0.0109), update (0.0096), every (0.0092), get (0.0090), download (0.0089), ca (0.0082), connection (0.0080), still (0.0079), work (0.0074), android (0.0070), problem (0.0069), using (0.0067), crash (0.0066), mode (0.0066), working (0.0065), error (0.0059), start (0.0058), connect (0.0057), back (0.0057)"
Tema_4,"free (0.0249), subscription (0.0209), ca (0.0208), even (0.0170), account (0.0166), log (0.0134), pay (0.0127), trial (0.0109), login (0.0107), sign (0.0104), service (0.0103), let (0.0100), sleep (0.0093), timer (0.0087), plan (0.0079), listen (0.0078), money (0.0078), get (0.0075), back (0.0074), wo (0.0073), month (0.0071), try (0.0069), dolby (0.0069), atmos (0.0065), want (0.0062), tried (0.0060), email (0.0060), say (0.0060), without (0.0060), card (0.0058)"
"""

data_neg = []

lines = texto_negativas.strip().split('\n')
for line in lines:
    tema_match = re.match(r'(Tema_\d+)', line)
    if tema_match:
        tema = tema_match.group(1)
        # Extraemos palabra y su probabilidad/peso
        matches = re.findall(r'(\w+)\s+\(([\d.]+)\)', line)
        for palabra, peso in matches:
            data_neg.append({
                'Aplicacion': 'TIDAL', # Cambia a 'SOUNDCLOUD' cuando proceses el otro
                'Sentimiento': 'Negativo',
                'Tema': tema,
                'Palabra': palabra,
                'Peso': float(peso)
            })

df_neg = pd.DataFrame(data_neg)
df_neg.to_excel('LDA_Tidal_Negativas.xlsx', index=False)

print("¡Archivo de Negativas generado para Tableau!")