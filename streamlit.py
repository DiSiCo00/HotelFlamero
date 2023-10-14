# -*- coding: utf-8 -*-
"""Streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hyOvsFySs9GuN2faampt6ddyLw9z1NJA
"""

#!pip install streamlit

#Importamos las librerías
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import random

import joblib

import warnings
warnings.filterwarnings('ignore' )

#pd.set_option('mode.chained_assignment', None)
while st.button('Reserva'):
    st.write("""
    ## REALIZA TU RESERVA AHORA
    """)

    hoy=st.date_input('¿Qué día es hoy?',
            min_value=pd.to_datetime(datetime.now()),
            max_value=pd.to_datetime('1/6/2024',dayfirst=True))

    #Recuperamos el modelo del random forest
    random_forest = joblib.load("random_forest.pkl")

    #Recuperamos el modelo del random forest
    random_forest_canc = joblib.load("random_forest_canc.pkl")

    #Leemos el csv reservas_total_preprocesado para recuperar el dataframe
    reservas_total=pd.read_csv('reservas_total_preprocesado.csv')

    #Convertimos las columnas en formato de fecha
    reservas_total['Fecha entrada'] = pd.to_datetime(reservas_total['Fecha entrada'], dayfirst=True)
    reservas_total['Fecha venta'] = pd.to_datetime(reservas_total['Fecha venta'], dayfirst=True)
    reservas_total['Fecha Anulacion'] = pd.to_datetime(reservas_total['Fecha Anulacion'], dayfirst=True, format='mixed')

    #Leemos el csv para recuperar el dataframe
    cancelaciones=pd.read_csv('cancelaciones.csv')

    #Función para predercir la probabilidad de cancelación de una reserva con un modelo determinado
    def predict_model(obj,model=random_forest):
      #Recuperamos las variables que participan en la predicción del modelo
      columnas_X=['Noches','Tip.Hab.Fra.','Régimen factura', 'AD', 'NI','CU','Horario venta',
            'Precio alojamiento','Precio desayuno', 'Precio almuerzo', 'Precio cena',
            'Cantidad Habitaciones','Mes Entrada','Mes Venta','Antelacion']

      #Tomamos nuestra base de entrenamiento para realizar el proceso de normalización y One Hot Encoding
      _sample = reservas_total[columnas_X]

      #Añadimos la nueva reserva a los datos
      X_new = pd.concat([_sample, pd.DataFrame(obj,index=[0])], ignore_index=True)

      #One Hot Encoding de las variables categóricas
      X_new = pd.get_dummies(X_new, columns=["Tip.Hab.Fra.", "Régimen factura","Horario venta", "Mes Entrada", "Mes Venta"], drop_first=True)

      #Aplicamos el escalador robusto
      robust_scaler = RobustScaler()
      X_new[["Precio alojamiento", "Antelacion"]] = robust_scaler.fit_transform(X_new[["Precio alojamiento", "Antelacion"]])

      #Aplicamos la normalización Min Max
      scaler = MinMaxScaler()
      X_new = scaler.fit_transform(X_new)

      prob=model.predict_proba(X_new[-1].reshape(1, -1))[0,1]

      #Predecimos la probabilidad de cancelación de la nueva reserva
      st.write(f"Su probabilidad de cancelación es de: {float(prob)}")
      return prob

    #Función para simular nuevas reservas aleatorias
    def random_Booking():
      #Elegimos un tipo de habitación y un régimen de nuestro dataset reservas_total al azar
      room_type=random.choice(reservas_total['Tip.Hab.Fra.'].unique())
      regimen=random.choice(reservas_total['Régimen factura'].unique())

      #Fijamos la hora actual como la hora de reserva
      hora = int(datetime.now().strftime('%H'))
      if (0 <= hora < 6):
        horario = 'Madrugada'
      elif 6 <= hora < 12:
        horario = 'Mañana'
      elif 12 <= hora < 18:
        horario = 'Tarde'
      else:
        horario = 'Noche'


      #Definimos el objeto con todas las variables necesarias
      obj = {
        "Noches": random.choice(reservas_total['Noches'].unique()) ,
        "Tip.Hab.Fra." : room_type ,
        "Régimen factura":regimen,
        "AD":random.choice(reservas_total['AD'].unique()),
        "NI":random.choice(reservas_total['NI'].unique()),
        "CU":random.choice(reservas_total['CU'].unique()),
        'Horario venta': horario,
        'Precio alojamiento': reservas_total['Precio alojamiento'].loc[reservas_total['Tip.Hab.Fra.'] == room_type].mean(),
        'Precio desayuno': reservas_total['Precio desayuno'].loc[reservas_total['Régimen factura'] == regimen].mean(),
        'Precio almuerzo': reservas_total['Precio almuerzo'].loc[reservas_total['Régimen factura'] == regimen].mean(),
        'Precio cena': reservas_total['Precio cena'].loc[reservas_total['Régimen factura'] == regimen].mean(),
        "Cantidad Habitaciones": int(reservas_total["Cantidad Habitaciones"].loc[reservas_total['Tip.Hab.Fra.'] == room_type].mean()),
        'Mes Entrada' : random.choice(reservas_total["Mes Entrada"].unique()),
        'Mes Venta': random.choice(reservas_total['Mes Venta'].unique()),
        'Antelacion': 300
      }

      return obj

    #Función para definir la cantidad mínima de habitaciones a reservar en base a huespedes y tipo de habitación
    def habitaciones(adultos, niños, tipo_habitacion):
      cont = 1

      #Si es una SUITE, la capacidad máxima es de 2 adultos y 2 niños o 3 adultos
      if tipo_habitacion == 'SUITE':
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2 + niños % 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 adultos
        if  adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es una habitación DELUXE VISTA COTO, la capacidad máxima es de 2 adultos y 1 niño
      if tipo_habitacion == 'DVC':
        #Si hay más niños que adultos devolvemos error (0)
        if adultos < niños:
          return 0

        #Asignamos una habitación por niño y 2 adultos por habitación
        cont = niños
        adultos -= cont * 2

        #Asignamos habitaciones de 2 adultos
        if  adultos > 0:
          cont += adultos // 2 + adultos % 2

      #Si es una habitación DELUXE VISTA MAR, la capacidad máxima es de 2 adultos. No se permiten niños
      if tipo_habitacion == 'DVM':
        #Si hay niños por adulto devolvemos error (0)
        if niños > 0:
          return 0

        #Asignamos habitaciones de 2 adultos
        cont = adultos // 2 + adultos % 2

      #Si es una habitación INDIVIDUAL, la capacidad máxima es de 1 adulto. No se permiten niños
      if tipo_habitacion == 'IND':
        #Si hay niños por adulto devolvemos error (0)
        if niños > 0:
          return 0

        #Asignamos las habitaciones individuales
        cont = adultos

      #Si es un APARTAMENTO PREMIUM, la capacidad máxima es de 4 adultos y 3 niños
      if tipo_habitacion == 'A':
        #Si hay más de 3 niños por adulto devolvemos error (0)
        if adultos * 3 < niños:
          return 0

        #Asignamos los niños de 3 en 3 y cuatro adultos por habitación
        cont = niños // 3
        niños = niños % 3
        adultos -= cont * 4

        #Si sobran niños asignamos otra habitación con capacidad para 4 adultos más
        if niños > 0:
          cont += 1
          adultos -= 4

        #Si sobran adultos asignamos habitaciones de 4 adultos
        if adultos > 0:
          cont += adultos // 4
          adultos = adultos % 4

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1

      #Si es un ESTUDIO estándar o una habitación DOBLE SUPERIOR, independientemente de si es vista COTO o MAR,
      #la capacidad máxima es de 3 adultos y 1 niño o 2 adultos y 2 niños
      if tipo_habitacion in ('EC', 'EM', 'DSC', 'DSM'):
        #Si hay más de 2 niños por adulto devolvemos error (0)
        if adultos * 2 < niños:
          return 0

        #Asignamos los niños de 2 en 2 y dos adultos por habitación
        cont = niños // 2
        adultos -= cont * 2

        #Asignamos habitaciones de 3 en 3
        if adultos > 0:
          cont += adultos // 3
          adultos = adultos % 3

          #Última habitación si sobran adultos
          if adultos > 0:
            cont += 1
        #Si no sobran adultos pero sí un niño, asignaremos una habitación extra
        elif niños % 2 == 1:
          cont += 1

      return cont

    #Función para crear nuevas reservas
    def new_Booking():
      reservas_total=pd.read_csv('reservas_total_preprocesado.csv')
    
      fecha_entrada=st.date_input('Introduzca la fecha de entrada:',
                value=pd.to_datetime('1/6/2024', dayfirst=True),
                min_value=pd.to_datetime('1/6/2024', dayfirst=True),
                max_value=pd.to_datetime('30/9/2024',dayfirst=True))

      noches=int(st.number_input('Seleccione la cantidad de noches:',min_value=1))

      print('Seleccione el número de adultos: \t')
      adultos=int(st.number_input('Seleccione el número de adultos:',min_value=1))

      print('Seleccione el número de niños: \t')
      niños=int(st.number_input('Seleccione el número de niños:',min_value=0))

      print('Seleccione el número de cunas: \t')
      cunas=int(st.number_input('Seleccione el número de cunas:',min_value=0))

      room_type=st.radio('Seleccione un tipo de habitación de entre los siguientes:',
                     ['DSC', 'DSM', 'DVC', 'DVM', 'EC', 'EM', 'IND', 'SUITE'])

      #Calculamos el número de habitaciones con la función
      num_habitaciones = habitaciones(adultos, niños, room_type)

      if (num_habitaciones == 0):
          st.write("La habitación no se adecúa a sus circunstancias. Seleccione otro tipo de habitación")
          return 0
      st.write(f"Usted va a reservar {num_habitaciones} habitaciones")
      regimen=st.radio('Seleccione un régimen de entre los siguientes:',
                  ['MPA', 'MPC','PC', 'HD', 'SA'])


      hora = int(datetime.now().strftime('%H'))
      if (0 <= hora < 6):
        horario = 'Madrugada'
      elif 6 <= hora < 12:
        horario = 'Mañana'
      elif 12 <= hora < 18:
        horario = 'Tarde'
      else:
        horario = 'Noche'

      precio_alojamiento=reservas_total['Precio alojamiento'].loc[reservas_total['Tip.Hab.Fra.'] == room_type].mean()
      precio_desayuno=reservas_total['Precio desayuno'].loc[reservas_total['Régimen factura'] == regimen].mean()
      precio_almuerzo=reservas_total['Precio almuerzo'].loc[reservas_total['Régimen factura'] == regimen].mean()
      precio_cena= reservas_total['Precio cena'].loc[reservas_total['Régimen factura'] == regimen].mean()
      precio_total=precio_alojamiento+precio_desayuno+precio_almuerzo+precio_cena

      obj = {
        "Noches": noches,
        "Tip.Hab.Fra." : room_type,
        "Régimen factura": regimen,
        "AD": adultos,
        "NI":niños,
        "CU":cunas,
        'Horario venta': horario,
        'Precio alojamiento': precio_alojamiento,
        'Precio desayuno': precio_desayuno,
        'Precio almuerzo': precio_almuerzo,
        'Precio cena': precio_cena,
        "Cantidad Habitaciones": num_habitaciones,
        'Mes Entrada' : fecha_entrada.strftime('%B'),
        'Mes Venta': datetime.now().strftime('%B'),
        'Antelacion': (fecha_entrada-hoy).days
      }

      return obj

    #Función para predecir la fecha de cancelación de la reserva
    def cancel_date(obj: dict,model_canc=random_forest_canc):
      #Definimos las variables que usaremos en el modelo
      columnas_canc_X = ['Noches', 'Tip.Hab.Fra.', 'Régimen factura', 'AD', 'NI', 'CU', 'Horario venta', 'Precio alojamiento', 'Precio desayuno',
                       'Precio almuerzo', 'Precio cena', 'Cantidad Habitaciones', 'Mes Entrada', 'Mes Venta', 'Antelacion']
      #Tomamos nuestra base de entrenamiento para realizar el proceso de normalización y One Hot Encoding
      _sample = cancelaciones[columnas_canc_X]

      #Añadimos la nueva reserva a los datos
      X_new = pd.concat([_sample, pd.DataFrame(obj,index=[0])], ignore_index=True)

      #One Hot Encoding de las variables categóricas
      X_new = pd.get_dummies(X_new, columns=["Tip.Hab.Fra.", "Régimen factura","Horario venta", "Mes Entrada", "Mes Venta"], drop_first=True)

      #Aplicamos el escalador robusto
      robust_scaler = RobustScaler()
      X_new[["Precio alojamiento", "Antelacion"]] = robust_scaler.fit_transform(X_new[["Precio alojamiento", "Antelacion"]])

      # Aplicamos la normalización Min Max
      X_norm = MinMaxScaler().fit_transform(X_new)

      #Predecimos el score con el modelo
      _score = model_canc.predict(X_norm[-1].reshape(1, -1))

      #Obtenemos los días que pasarán hasta la cancelación
      _days = float(_score) * obj["Antelacion"]

      #Sumamos los días a la fecha actual
      cancel_date = hoy + timedelta(_days)

      st.write(f"La reserva se podría cancelar el día {cancel_date}")
      return cancel_date
    
    booking_date=new_Booking()
    if booking_date != 0:
        predict_model(booking_date)
        cancel_date(booking_date)
