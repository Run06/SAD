from ollama import chat
import csv
import random
import pandas as pd

csvfile = pd.read_csv('TIDAL.csv')
neg = []
neu = []
pos = []
tot = 0
n = 2 #Hiperparametro cuantos ejemplos por categoria
while tot < 3*n:
    linea = csvfile.sample()
    score = int(linea['score'].values[0])
    if(score <= 2):
        if len(neg) < n:
            neg.append(linea['review'].values[0])
            tot = tot+1
    elif(score >= 4):
        if len(pos) < n:
            pos.append(linea['review'].values[0])
            tot = tot+1
    elif(score == 3):
        if len(neu) < n:
            neu.append(linea['review'].values[0])
            tot = tot+1

messages = [
    {
        "role": "user",
        "content": "Paraphrase:",
    }, 
    {
        "role": "user",
        "content": "The app doesn't work well => The applications works bad",
    }
]

mess_test = csvfile.sample()
print(mess_test['review'].values[0])

mess = {
    "role": "user",
    "content": mess_test['review'].values[0] + " =>",
}
messages.append(mess)

response = chat(model="llama3:8b-text-q2_K", messages=messages)
print("-----------Respuesta de Ollama:--------------")
print(response.message.content)
