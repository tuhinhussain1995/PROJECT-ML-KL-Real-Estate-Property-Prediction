from django.shortcuts import render
from django.http import HttpResponse



def index(request):
    return render(request, 'index.html')



import numpy as np
import pandas as pd
import pickle
import json

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pkldir = os.path.join(BASE_DIR, 'static/ml_files/malaysia_property_price_final.pickle')
jsondir = os.path.join(BASE_DIR, 'static/ml_files/malaysia_property_price_final_columns.json')


def register(request):
    if request.method == 'POST':
        Squareft = request.POST['Squareft']
        uiBHK = request.POST['uiBHK']
        uiBathrooms = request.POST['uiBathrooms']
        carparking = request.POST['carparking']
        type = request.POST['type']
        furnishing = request.POST['furnishing']
        location = request.POST['location']

        with open(pkldir, 'rb') as f:
            mp = pickle.load(f)

        with open(jsondir) as json_file:
            XX = json.load(json_file)


        val1 = predict_price(location, uiBHK, uiBathrooms, carparking, Squareft, type, furnishing)

        my_value = int(np.round(val1))
        f = str(my_value)

        inc = 0
        mylist = []

        for x in range(len(f)-1, -1, -1):
            mylist.append(f[x])

            inc += 1

            if inc == 3:
                mylist.append(',')
                inc = 0

        mylist.reverse()
        ans = ''.join(mylist)

        if ans[0] == ',':
            ans = ans[1:]

        final_ans = 'RM ' + ans

    return render(request, 'index.html', {'number1' : final_ans})




def predict_price(location, room, bathroom, car, size, typ, furn):

    with open(pkldir, 'rb') as f:
        mp = pickle.load(f)

    with open(jsondir) as json_file:
        XX = json.load(json_file)

    loc_index = 0
    typ_index = 0
    furn_index = 0

    for i in range(len(XX['data_columns'])):
        if XX['data_columns'][i] == location:
            loc_index = i
            break

    for i in range(len(XX['data_columns'])):
        if XX['data_columns'][i] == typ:
            typ_index = i
            break

    for i in range(len(XX['data_columns'])):
        if XX['data_columns'][i] == furn:
            furn_index = i
            break

    x = np.zeros(len(XX['data_columns']))
    x[0] = room
    x[1] = bathroom
    x[2] = car
    x[3] = size

    if loc_index >= 0:
        x[loc_index] = 1

    if typ_index >= 0:
        x[typ_index] = 1

    if furn_index >= 0:
        x[furn_index] = 1

    return mp.predict([x])[0]
