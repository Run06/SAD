from ollama import chat
import argparse
import csv
import random
import pandas as pd

parse = argparse.ArgumentParser(description="Analisis de sentimientos por IA")
parse.add_argument("-f", "--file", help="Fichero csv", required=True)
parse.add_argument("-n", "--number", help="Numero de parafrasis", required=True)
args = parse.parse_args()

csvfile = pd.read_csv(args.file)


contexto = [
    {
        "role": "user",
        "content": "Paraphrase:",
    }, 
    {
        "role": "user",
        "content": "The app doesn't work well => The applications works bad",
    }
]

i = 0
respuestas = []
while i < int(args.number):
    i = i + 1
    print(i)
    messages = []
    for m in contexto:
        messages.append(m)
    mess_test = csvfile.sample()
    print(mess_test['review'].values[0])
    mess = {
        "role": "user",
        "content": mess_test['review'].values[0] + " =>",
    }
    
    messages.append(mess)

    response = chat(model="llama3:8b-text-q2_K", messages=messages)

    res = {'n_review': str(response.message.content).split('>'), 'score': mess_test['score'].values[0]}
    respuestas.append(res)
    print(response.message.content)
    
        

with open('para.csv', 'w', newline='') as csvfile:
    fieldnames = ['n_review', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(respuestas)
print("hecho :3")
