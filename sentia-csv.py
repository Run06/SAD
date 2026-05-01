from ollama import chat
import csv
import random
import pandas as pd
import argparse



parse = argparse.ArgumentParser(description="Analisis de sentimientos por IA")
parse.add_argument("-f", "--file", help="Fichero csv", required=True)
parse.add_argument("-s", "--shots", help="Numero de ejemplos por categoría", required=True)
parse.add_argument("-n", "--number", help="Numero de filas del csv", required=True)

args = parse.parse_args()
csvfile = pd.read_csv(args.file)


neg = []
neu = []
pos = []
tot = 0
n = int(args.shots) #Hiperparametro cuantos ejemplos por categoria
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

context = [
    {
        "role": "user",
        "content": "/clear",
    }, 
    {
        "role": "user",
        "content": "Perform sentiment analysis:",
    }
]

for review in pos:
    mess = {
                "role": "user",
                "content": review + " => Positive",
            }
    context.append(mess)
for review in neu:
    mess = {
                "role": "user",
                "content": review + " => Neutral",
            }
    context.append(mess)
for review in neg:
    mess = {
                "role": "user",
                "content": review + " => Negative",
            }
    context.append(mess)

i = 0

respuestas = []
while i < int(args.number):
    i = i + 1
    print(i)
    df = pd.read_csv(args.file)
    messages = []
    for m in context:
        messages.append(m)
    mess_test = df.sample()
    mess = {
        "role": "user",
        "content": mess_test['review'].values[0] + " =>",
    }
    messages.append(mess)
    
    response = chat(model="llama3:8b-text-q2_K", messages=messages, options={"num_predict": 1})
    res = {'review': mess_test['review'].values[0], 'sentimiento' : str(response.message.content), 'score': mess_test['score'].values[0]}
    respuestas.append(res)
    print(response.message.content)
    
        

with open('sent.csv', 'w', newline='') as csvfile:
    fieldnames = ['review', 'sentimiento', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(respuestas)

mtrxConf = ([[0,0,0],
             [0,0,0], 
             [0,0,0]])

with open('sent.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        scoreReal = int(row['score'])
        if(scoreReal <= 2):
            if 'Negative' in row['sentimiento']:
                mtrxConf[0][0] = mtrxConf[0][0] + 1
            elif 'Neutral' in row['sentimiento']:
                mtrxConf[0][1] = mtrxConf[0][1] + 1
            elif 'Positive' in row['sentimiento']:
                mtrxConf[0][1] = mtrxConf[0][2] + 1
        elif(scoreReal >= 4):
            if 'Negative' in row['sentimiento']:
                mtrxConf[2][0] = mtrxConf[2][0] + 1
            elif 'Neutral' in row['sentimiento']:
                mtrxConf[2][1] = mtrxConf[2][1] + 1
            elif 'Positive' in row['sentimiento']:
                mtrxConf[2][2] = mtrxConf[2][2] + 1
        elif(scoreReal == 3):
            if 'Negative' in row['sentimiento']:
                mtrxConf[1][0] = mtrxConf[1][0] + 1
            elif 'Neutral' in row['sentimiento']:
                mtrxConf[1][1] = mtrxConf[1][1] + 1
            elif 'Positive' in row['sentimiento']:
                mtrxConf[1][2] = mtrxConf[1][2] + 1
print('')
print('Matriz de confusión:')
for row in mtrxConf:
    print(row)
print('')


precNeg = mtrxConf[0][0]/(mtrxConf[0][0] + mtrxConf[0][1] + mtrxConf[0][2])
recNeg = mtrxConf[0][0]/(mtrxConf[0][0] + mtrxConf[1][0] + mtrxConf[2][0])
fScoreNeg = (2 * precNeg * recNeg)/(precNeg+recNeg)
print('Negative:')
print('F-Score: ' + str(fScoreNeg))
print('Precision: ' + str(precNeg))
print('Recall: ' + str(recNeg))
print('')

precNeu = mtrxConf[1][1]/(mtrxConf[1][0] + mtrxConf[1][1] + mtrxConf[1][2])
recNeu = mtrxConf[1][1]/(mtrxConf[0][1] + mtrxConf[1][1] + mtrxConf[2][1])
fScoreNeu = (2 * precNeu * recNeu)/(precNeu+recNeu)
print('Neutral:')
print('F-Score: ' + str(fScoreNeu))
print('Precision: ' + str(precNeu))
print('Recall: ' + str(recNeu))
print('')

precPos = mtrxConf[2][2]/(mtrxConf[2][0] + mtrxConf[2][1] + mtrxConf[2][2])
recPos = mtrxConf[2][2]/(mtrxConf[2][0] + mtrxConf[2][1] + mtrxConf[2][2])
fScorePos = (2 * precPos * recPos)/(precPos+recPos)
print('Positive:')
print('F-Score: ' + str(fScorePos))
print('Precision: ' + str(precPos))
print('Recall: ' + str(recPos))
print('')

macroFScore = (fScoreNeg + fScoreNeu + fScorePos)/3
print('Macro F-Score: ' + str(macroFScore))

