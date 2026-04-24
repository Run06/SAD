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
with open('sent.csv', 'w', newline='') as csvfile:
    fieldnames = ['review', 'sentimiento', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

while i < int(args.number):
    i = i + 1
    print(i)
    df = pd.read_csv(args.file)
    messages = context
    mess_test = df.sample()
    mess = {
        "role": "user",
        "content": mess_test['review'].values[0] + " =>",
    }
    messages.append(mess)
    
    response = chat(model="llama3:8b-text-q2_K", messages=messages, options={"num_predict": 1})
    with open('sent.csv', 'w', newline='') as csvfile:
        fieldnames = ['review', 'sentimiento', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'review': mess_test['review'].values[0], 'sentimiento' : str(response.message.content), 'score': mess_test['score'].values[0]})


print("hecho :3")

