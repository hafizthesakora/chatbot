from fastapi import FastAPI
import uvicorn
import numpy as np
import os
import pickle
import nltk
import json
import random

from nltk.stem import WordNetLemmatizer
from answer import Answer
from question import Question

from tensorflow.keras.models import load_model

app = FastAPI()

MODEL = load_model(os.path.join(os.curdir,'chatbot_model.h5'))

WORDS = pickle.load(open('words.pkl','rb'))
CLASSES = pickle.load(open('classes.pkl','rb'))

lemmatizer = WordNetLemmatizer()



def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def get_numerical_representation_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    result = [0] * len(WORDS)
    for word in sentence_words:
        for index,w in enumerate(WORDS):
            if word == w:
                result[index] = 1
    return np.array(result)


def predict(sentence):
    rep = get_numerical_representation_of_words(sentence)
    prediction = MODEL.predict(np.array([rep]))[0]
    THRESHOLD  = 0.25
    results = [[i,r] for i,r in enumerate(prediction) if r > THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({ 'intent': CLASSES[r[0]] , 'probability' : str(r[1]) })
    return return_list 

def get_response(predicted_class):
    with open('./data/Intent.json','rb') as file:
        intents = json.loads(file.read())['intents']
        response = None
        for intent in intents:
            if intent['intent'] ==  predicted_class:
                response = random.choice(intent['responses'])
                break 
        return response

@app.post('/question')
async def get_answer(data:Question):
    sentence = data.question; 
    response = get_response(predict(sentence)[0]['intent'])
    return Answer(response=response)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.environ.get("PORT",8000))
