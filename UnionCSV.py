import pandas as pd
import re

# 1. PEGA AQUÍ TUS INFORMES
texto_positivas = """
Tema_0,"better (0.0715), artist (0.0494), awesome (0.0296), much (0.0261), pay (0.0193), way (0.0150), selection (0.0147), get (0.0126), streaming (0.0121), apps (0.0111), listen (0.0110), audio (0.0106), well (0.0104), sound (0.0098), free (0.0097), perfect (0.0096), apple (0.0093), ui (0.0090), new (0.0087), favorite (0.0069), top (0.0069), exceptional (0.0064), premium (0.0063), super (0.0062), frustrating (0.0061), actually (0.0059), recommend (0.0059), hifi (0.0058), guy (0.0055), pretty (0.0054)"
Tema_1,"excellent (0.0222), nice (0.0164), update (0.0150), issue (0.0117), hifi (0.0107), phone (0.0105), work (0.0104), high (0.0095), audio (0.0094), time (0.0088), using (0.0079), really (0.0077), new (0.0073), thing (0.0073), keep (0.0073), play (0.0070), device (0.0067), would (0.0064), well (0.0064), used (0.0064), back (0.0061), also (0.0060), year (0.0059), system (0.0059), price (0.0058), home (0.0054), support (0.0053), android (0.0053), headphone (0.0053), service (0.0052)"
Tema_2,"amazing (0.0260), song (0.0248), experience (0.0203), make (0.0182), playlist (0.0158), really (0.0153), one (0.0140), could (0.0133), listen (0.0122), feature (0.0113), many (0.0106), using (0.0099), want (0.0097), please (0.0094), would (0.0094), get (0.0088), day (0.0086), every (0.0085), absolutely (0.0084), album (0.0082), add (0.0081), listening (0.0078), feel (0.0077), know (0.0075), definitely (0.0074), subscription (0.0068), less (0.0067), new (0.0066), improved (0.0065), year (0.0064)"
Tema_3,"streaming (0.0533), service (0.0436), audio (0.0272), easy (0.0200), interface (0.0166), user (0.0156), always (0.0132), search (0.0128), find (0.0121), need (0.0120), platform (0.0119), artist (0.0089), experience (0.0082), thanks (0.0080), playlist (0.0079), also (0.0078), lot (0.0065), high (0.0063), make (0.0059), way (0.0058), option (0.0056), everything (0.0055), one (0.0055), master (0.0055), superior (0.0054), content (0.0054), looking (0.0053), simply (0.0050), appealing (0.0050), hifi (0.0049)"

"""

texto_negativas = """
Tema_0,"playlist (0.0369), song (0.0261), artist (0.0175), track (0.0153), album (0.0152), play (0.0120), also (0.0111), add (0.0106), would (0.0103), one (0.0099), please (0.0098), want (0.0084), option (0.0081), feature (0.0080), ca (0.0075), phone (0.0074), search (0.0072), android (0.0071), make (0.0070), really (0.0066), new (0.0063), shuffle (0.0060), listen (0.0059), back (0.0058), way (0.0058), need (0.0057), thing (0.0056), go (0.0054), find (0.0054), update (0.0053)"
Tema_1,"make (0.0311), experience (0.0271), user (0.0239), interface (0.0236), less (0.0183), feel (0.0182), using (0.0165), android (0.0162), audio (0.0162), many (0.0145), watch (0.0141), playback (0.0141), auto (0.0135), saved (0.0120), thing (0.0103), issue (0.0103), get (0.0100), slow (0.0097), friendly (0.0092), recommendation (0.0089), samsung (0.0087), problem (0.0086), always (0.0086), frustrating (0.0086), know (0.0085), people (0.0084), galaxy (0.0079), occasionally (0.0074), generally (0.0072), one (0.0071)"
Tema_2,"better (0.0272), audio (0.0186), service (0.0162), would (0.0136), streaming (0.0136), need (0.0124), could (0.0124), artist (0.0108), feature (0.0093), google (0.0091), much (0.0089), really (0.0089), search (0.0082), also (0.0080), high (0.0076), find (0.0073), bit (0.0071), hifi (0.0068), star (0.0067), still (0.0066), integration (0.0065), available (0.0063), ui (0.0062), master (0.0061), wish (0.0061), nice (0.0058), pretty (0.0057), lot (0.0057), overall (0.0056), selection (0.0056)"
Tema_3,"play (0.0192), issue (0.0178), time (0.0177), downloaded (0.0148), playing (0.0138), stop (0.0134), offline (0.0133), keep (0.0123), fix (0.0118), phone (0.0117), song (0.0115), even (0.0109), update (0.0096), every (0.0092), get (0.0090), download (0.0089), ca (0.0082), connection (0.0080), still (0.0079), work (0.0074), android (0.0070), problem (0.0069), using (0.0067), crash (0.0066), mode (0.0066), working (0.0065), error (0.0059), start (0.0058), connect (0.0057), back (0.0057)"
Tema_4,"free (0.0249), subscription (0.0209), ca (0.0208), even (0.0170), account (0.0166), log (0.0134), pay (0.0127), trial (0.0109), login (0.0107), sign (0.0104), service (0.0103), let (0.0100), sleep (0.0093), timer (0.0087), plan (0.0079), listen (0.0078), money (0.0078), get (0.0075), back (0.0074), wo (0.0073), month (0.0071), try (0.0069), dolby (0.0069), atmos (0.0065), want (0.0062), tried (0.0060), email (0.0060), say (0.0060), without (0.0060), card (0.0058)"

"""

def procesar_lda(texto, sentimiento):
    data = []
    lines = texto.strip().split('\n')
    for line in lines:
        tema_match = re.match(r'(Tema_\d+)', line)
        if tema_match:
            tema = tema_match.group(1)
            # Extraer palabra y peso
            matches = re.findall(r'([\w\s]+)\s+\(([\d.]+)\)', line)
            for palabra, peso in matches:
                data.append({
                    'Aplicacion': 'TIDAL',
                    'Sentimiento': sentimiento,
                    'Tema': tema,
                    'Palabra': palabra.strip(),
                    'Peso': float(peso)
                })
    return data

# 2. PROCESAR Y JUNTAR
datos_totales = procesar_lda(texto_positivas, 'Positivo') + procesar_lda(texto_negativas, 'Negativo')

# 3. GUARDAR PARA TABLEAU
df = pd.DataFrame(datos_totales)
df.to_excel('LDA_TIDAL_COMPLETO.xlsx', index=False)

print("¡Hecho! Archivo 'LDA_TIDAL_COMPLETO.xlsx' generado para Tableau.")