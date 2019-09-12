import requests
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import re
import os
from datetime import date

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download("stopwords")
nltk.download("punkt")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BOT_TOKEN = '<YOUR-TELEGRAM-BOT-TOKEN>'
bot = "https://api.telegram.org/bot"+BOT_TOKEN+"/"


MET_TOKEN = '<MET-MALAYSIA-API-TOKEN>'
BASE_URL = 'https://api.met.gov.my/v2/'
headers = {'Authorization': 'METToken ' + MET_TOKEN}

def get_weather(LOC):
    today = date.today()
    current_date = today.strftime("%Y-%m-%d")
    response = requests.get(BASE_URL+"data?datasetid=FORECAST&datacategoryid=GENERAL&locationid=LOCATION:"+str(LOC)+"&start_date="+current_date+"&end_date="+current_date, headers=headers)
    result = response.json()['results']
    for weather in result:
        if weather['datatype'] == 'FGM':
            morning = weather['value']
        if weather['datatype'] == 'FGA':
            afternoon = weather['value']
        if weather['datatype'] == 'FGN':
            night = weather['value']
        if weather['datatype'] == 'FMAXT':
            temp_max = weather['value']
        if weather['datatype'] == 'FMINT':
            temp_min = weather['value']
    return 'Morning = ' + morning + ', Afternoon = '+ afternoon + ', Night = '+ night + ', Max Temp = '+ str(temp_max) + ', Min Temp = '+ str(temp_min)

def cleaning(sentence):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", sentence)
    w = word_tokenize(clean)
    clean_words = [i.lower() for i in w]
    return clean_words

def create_token(sentence):
    token = tf.keras.preprocessing.text.Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    token.fit_on_texts(sentence)
    return token

def get_intent(text, max_length, classes):
    test_word = cleaning(text)
    with open("cleaned_words.txt", "rb") as fp:  
        cleaned_words = pickle.load(fp)
    test_tokenizer = create_token(cleaned_words)
    test_ls = test_tokenizer.texts_to_sequences(test_word)
    print(test_word)
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls)) 
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = tf.keras.preprocessing.sequence.pad_sequences(test_ls, maxlen = max_length, padding = "post")
    model = tf.keras.models.load_model('model.h5')
    pred = model.predict_proba(x)
    predictions = pred[0]
    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)
    return classes[0], predictions[0]

def get_map_url(latitude,longitude):
    return 'https://www.google.com/maps/search/?api=1&query='+str(latitude)+','+str(longitude)

def get_info(df, place_name):
    info = []
    index = next(iter(df[df['Name']==place_name].index), 200)
    if index < 200:
        info = [df['Type'][index], df['District'][index], df['Latitude'][index], df['Longitude'][index], df['Review'][index], df['Telephone'][index], df['Operation Hour'][index], df['Entrance Fee'][index], df['Category'][index]]
    else:
        info = []
    return info

def get_response(dataframe, name_list, district_list, input_text):
    # intent classification
    max_length = 7
    intents = ['direction', 'operation', 'review', 'stay', 'fee', 'place', 'weather', 'phone', 'eat']
    intent, confidence = get_intent(input_text, max_length, intents)
    print("%s with confidence = %s" % (intent, confidence))

    # entity recognition
    place = ''
    for district in district_list:
        cleaned_district = TreebankWordDetokenizer().detokenize(cleaning(district))
        if cleaned_district in TreebankWordDetokenizer().detokenize(cleaning(input_text)):
            place = district
    for name in name_list:
        cleaned_name = TreebankWordDetokenizer().detokenize(cleaning(name))
        if cleaned_name in TreebankWordDetokenizer().detokenize(cleaning(input_text)):
            place = name
    print('Place recognized = ', place)

    # Main logic
    if intent == 'direction':
        if place in name_list:
            place_info = get_info(dataframe, place)
            map_url = get_map_url(place_info[2],place_info[3])
            response = map_url
        else:
            response = 'Invalid place'
    elif intent == 'review':
        if place in name_list:
            place_info = get_info(dataframe, place)
            response = 'Based on Google review, the place is rated ' + str(place_info[4]) + '/5' 
        else:
            response = 'Invalid place'
    elif intent == 'weather':
        if place in name_list:
            place_info = get_info(dataframe, place)
            if place_info[1] == 'Jeli ':
                loc = 40
            elif place_info[1] == 'Bachok':
                loc = 38
            elif place_info[1] == 'Pasir Mas':
                loc = 45
            elif place_info[1] == 'Tanah merah':
                loc = 47
            elif place_info[1] == 'Gua Musang':
                loc = 39
            elif place_info[1] == 'Tumpat':
                loc = 48
            elif place_info[1] == 'Kuala Krai':
                loc = 42
            elif place_info[1] == 'Machang':
                loc = 44
            elif place_info[1] == 'Pasir Puteh':
                loc = 46
            elif place_info[1] == 'Kota Bharu':
                loc = 41
            response = get_weather(loc)
        else:
            response = 'Invalid place'
    elif intent == 'phone':
        if place in name_list:
            place_info = get_info(dataframe, place)
            response = 'The phone number is ' + str(place_info[5])
        else:
            response = 'Invalid place'
    elif intent == 'operation':
        if place in name_list:
            place_info = get_info(dataframe, place)
            response = 'The operation hour is ' + str(place_info[6])
        else:
            print('Invalid place')
    elif intent == 'fee':
        if place in name_list:
            place_info = get_info(dataframe, place)
            response = 'The entrance fee is ' + str(place_info[7])
        else:
            response = 'Invalid place'
    elif intent == 'eat':
        if place in district_list:
            indexes = dataframe[dataframe['Type']=='Eat'].index & dataframe[dataframe['District']==place].index
            response_list = ''
            if len(indexes) > 0:
                for i in indexes:
                    response_list = response_list + dataframe['Name'][i] + ' - ' + dataframe['Category'][i] + '\n'
                response = 'Nice restaurants for you to fill your tummy:\n' + response_list
            else:
                response_list = 'Sorry! no place found'
        else:
            response = 'Invalid place'
    elif intent == 'place':
        if place in district_list:
            indexes = dataframe[dataframe['Type']=='Visit'].index & dataframe[dataframe['District']==place].index
            response_list = ''
            if len(indexes) > 0:
                for i in indexes:
                    response_list = response_list + dataframe['Name'][i] + ' - ' + dataframe['Category'][i] + '\n'
                response = 'Amzing place to visit:\n' + response_list
            else:
                response_list = 'Sorry! no place found'
        else:
            response = 'Invalid place'
    elif intent == 'stay':
        if place in district_list:
            indexes = dataframe[dataframe['Type']=='Stay'].index & dataframe[dataframe['District']==place].index
            response_list = ''
            if len(indexes) > 0:
                for i in indexes:
                    response_list = response_list + dataframe['Name'][i] + '\n'
                response = 'Nice place for you to spend the night:\n' + response_list
            else:
                response = 'Sorry! no place found'
            
        else:
            response = 'Invalid place'
    return response

def get_message():
    params = {'timeout': 100, 'offset': None}
    response = requests.get(bot + 'getUpdates', data=params)
    results = response.json()['result']
    latest_update = len(results) - 1
    chat_id =  results[latest_update]['message']['chat']['id']
    update = results[latest_update]['update_id']
    if 'text' in results[latest_update]['message'].keys():
        text =  results[latest_update]['message']['text'] 
    else:
        text = 'Invalid input type'
    return chat_id, text, update

def send_message(chat, reply_text):
    params = {'chat_id': chat, 'text': reply_text}
    response = requests.post(bot + 'sendMessage', data=params)
    return response

def main():
    last_update_id = 0
    dataframe = pd.read_csv('database.csv')
    name_list = list(set(dataframe['Name']))
    district_list = list(set(dataframe['District']))
    while True:
        chat_id, input_text, update_id = get_message()
        if update_id > last_update_id: 
            if input_text == '/start':
                reply = 'Hi! I\'m TourMate. How I may assist you?'
                send_message(chat_id, reply)
            elif input_text == 'Invalid input type':
                reply = input_text
                send_message(chat_id, reply)
            else:
                reply = get_response(dataframe, name_list, district_list, input_text)
                print(reply)
                K.clear_session()
                send_message(chat_id, reply)
            last_update_id = update_id
        else:
            continue

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()