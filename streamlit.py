# -*- coding: utf-8 -*-
"""Streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hyOvsFySs9GuN2faampt6ddyLw9z1NJA
"""
#install streamlit-option-menu

#Importamos las librerías
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')
import openai, os, requests

import nltk
from translate import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
nltk.download('punkt')

with st.sidebar:
    selected = option_menu('Menu', ['Chatbot','Reserva','Reseñas'])

#pd.set_option('mode.chained_assignment', None)

if selected == 0:
    openai.api_type = "azure"

    # Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
    openai.api_version = "2023-08-01-preview"

    # Azure OpenAI setup
    openai.api_base = "https://chatbotflamero.openai.azure.com/openai/deployments/chatbot_flamero/extensions/chat/completions?api-version=2023-07-01-preview" # Add your endpoint here
    openai.api_key = os.getenv("ff4e5db40bb04cdca3e4d62afa342369") # Add your OpenAI API key here
    deployment_id = "chatbot_flamero" # Add your deployment ID here
    # Azure Cognitive Search setup
    search_endpoint = "https://chatbotflamero.search.windows.net"; # Add your Azure Cognitive Search endpoint here
    search_key = Environment.GetEnvironmentVariable("4BCsu5yM5VNcjZaGSSB51GEWXAjLvwrUy3d0fS0J2RAzSeBUkCIh"); # Add your Azure Cognitive Search admin key here
    search_index_name = "flameroindex"; # Add your Azure Cognitive Search index name here
    
    def setup_byod(deployment_id: str) -> None:
        """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.
    
    :param deployment_id: The deployment ID for the model to use with your own data.

    To remove this configuration, simply set openai.requestssession to None.
    """

        class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

            def send(self, request, **kwargs):
                request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
                return super().send(request, **kwargs)

        session = requests.Session()

        # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
        session.mount(
            prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
            adapter=BringYourOwnDataAdapter()
    )

        openai.requestssession = session

    setup_byod(deployment_id)

    completion = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": "What are the differences between Azure Machine Learning and Azure AI services?"}],
        deployment_id=deployment_id,
        dataSources=[  # camelCase is intentional, as this is the format the API expects
        {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": search_endpoint,
                    "key": search_key,
                    "indexName": search_index_name,
                }
            }
        ]
    )
    st.write(completion)


if selected == 'Reserva':#st.button('Reserva'):
    st.write("""
    ## REALIZA TU RESERVA AHORA
    """)

    hoy=st.date_input('¿Qué día es hoy?',
            min_value=pd.to_datetime(datetime.now()),
            max_value=pd.to_datetime('31/8/2024',dayfirst=True))

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
      if (fecha_entrada<hoy):
          st.write('Introduzca una fecha de entrada posterior al día de hoy')
          return 0
      noches=int(st.number_input('Seleccione la cantidad de noches:',min_value=1))

      adultos=int(st.number_input('Seleccione el número de adultos:',min_value=1))

      niños=int(st.number_input('Seleccione el número de niños:',min_value=0))

      cunas=int(st.number_input('Seleccione el número de cunas:',min_value=0))

      room_type=st.radio('Seleccione un tipo de habitación de entre los siguientes:',
                     ['DSC', 'DSM', 'DVC', 'DVM', 'EC', 'EM', 'IND', 'SUITE', 'A'],
                     captions=['DOBLE SUPERIOR COTO', 'DOBLE SUPERIOR MAR', 'DELUXE VISTA COTO', 'DELUXE VISTA MAR', 
                               'ESTUDIO COTO', 'ESTUDIO MAR', 'INDIVIDUAL', 'SUITE', 'APARTAMENTO PREMIUM'])

      #Calculamos el número de habitaciones con la función
      num_habitaciones = habitaciones(adultos, niños, room_type)

      if (num_habitaciones == 0):
          st.write("La habitación no se adecúa a sus circunstancias. Seleccione otro tipo de habitación")
          return 0
      st.write(f"Usted va a reservar {num_habitaciones} habitaciones")
      regimen=st.radio('Seleccione un régimen de entre los siguientes:',
                  ['MPA', 'MPC','PC', 'HD', 'SA'],
                  captions=['MEDIA PENSIÓN ALMUERZO','MEDIA PENSIÓN CENA', 'PENSIÓN COMPLETA', 'HABITACIÓN Y DESAYUNO', 'SOLO ALOJAMIENTO'])


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
        'Mes Venta': hoy.strftime('%B'),
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

      if _score < 0.5:
          st.write(f"¡¡Aviso de posible cancelación tardía!!")
      return _score

    #Función cuota no reembolsable
    def func_no_reembolso(_obj, _cuota_media=0.25, _cuota_maxima=0.5, _umbral_inferior=0.25, _umbral_superior=0.4, model=random_forest, model_canc=random_forest_canc):

        if 0 <= _cuota_maxima <= 1:
          if 0 <= _cuota_media <= 1:
            if 0 <= _umbral_inferior <= 1:
              if 0 <= _umbral_superior <= 1:
                if _umbral_superior >_umbral_inferior:

                  _pred = predict_model(_obj, model)

                  if _pred < _umbral_inferior:
                    st.write("La nueva reserva tiene bajo riesgo de cancelación. Cancelación gratuita.")
                  elif _pred > _umbral_superior:
                    st.write(f"Alto Riesgo de cancelación. Aplicar {(_cuota_maxima)*100:.1f}% del Precio total.")
                    cancel_date(_obj, model_canc)
                  else:
                    st.write(f"Riesgo Moderado. Aplicar el {float((_cuota_media)*100):.1f}% del precio total. \n")
                    cancel_date(_obj, model_canc)
                else:
                  raise ValueError("El valor de ´umbral_superior´  tiene que ser mayor que ´umbral_inferior´.")
              else:
                raise ValueError("El valor ´umbral_superior´ debe estar entre 0 y 1.")
            else:
              raise ValueError("El valor ´umbral_inferior´ debe estar entre 0 y 1.")
          else:
             raise ValueError("El valor ´cuota_media´ debe estar entre 0 y 1.")
        else:
          raise ValueError("El valor ´cuota_maxima´ debe estar entre 0 y 1.")
    
    booking=new_Booking()
    if booking != 0:
        func_no_reembolso(booking)
        
        
if selected == 'Reseñas':

    reseña=st.text_area('Reseña:')
    if len(reseña) != 0:
        sia = SentimentIntensityAnalyzer()

        translator = Translator(from_lang="es", to_lang="en")
        reseña = translator.translate(reseña)


        categoria_limpieza = ["clean","tidy", "dirt", "cleaner", "towels", "rat", "rats", "cockroach", "cockroachs", "cleaner", "cleaners"]
        categoria_instalaciones = ["pool","elevator","buffet", "lobby", "gym", "door", "doors", "water", "parking","facilities"]
        categoria_habitacion = ["room", "rooms", "suite", "suites","bathroom", "toilet", "bedroom", "bedrooms", "towels", "roomy", "spacious", "bright", "luminous"]
        categoria_ubicacion = ["location","place","views", "beach", "sea", "preserve", "reserve"]
        categoria_atencion = ["needs", "requirements", "staff", "reception", "support", "help", "care", "careless", "careful", "attendance", "gentle", "charm", "helpful", "attentive", "receptionist", "waiter", "waiters", "cleaner", "cleaners", "service", "employees", "friendly", "unfriendly", "kind", "kindly"]
        categoria_tranquilidad = ["quiet", "noise", "noisy", "relax", "chill", "privacy"]
        categoria_comida = ["food", "buffet", "breakfast", "dinner", "lunch", "fruit", "fruits", "gastronomy", "hungry", "meat", "fish", "bread", "delicious"]
        categoria_precio = ["cheap", "expensive", "money", "savings", "economic"]
    
        def calcular_categoria_sentimiento(frase_ingles):
            tokens = nltk.word_tokenize(frase_ingles)
            categorias=[]
            cuenta_ubicacion = 0
            cuenta_habitacion = 0
            cuenta_limpieza = 0
            cuenta_instalaciones = 0
            cuenta_atencion = 0
            cuenta_tranquilidad = 0
            cuenta_comida = 0
            cuenta_precio = 0 
        
            for token in tokens:
                if token in categoria_ubicacion and cuenta_ubicacion == 0:
                    categorias.append('Ubicación')
                    cuenta_ubicacion = 1
                elif token in categoria_habitacion and cuenta_habitacion == 0:
                    categorias.append('Habitación')
                    cuenta_habitacion = 1
                elif token in categoria_limpieza and cuenta_limpieza == 0:
                    categorias.append('Limpieza')
                    cuenta_limpieza = 1
                elif token in categoria_instalaciones and cuenta_instalaciones == 0:
                    categorias.append('Instalaciones')
                    cuenta_instalaciones = 1
                elif token in categoria_atencion and cuenta_atencion == 0:
                    categorias.append('Atención al cliente')
                    cuenta_atencion = 1
                elif token in categoria_tranquilidad and cuenta_tranquilidad == 0:
                    categorias.append('Tranquilidad')
                    cuenta_tranquilidad = 1
                elif token in categoria_comida and cuenta_comida == 0:
                    categorias.append('Comida')
                    cuenta_comida = 1
                elif token in categoria_precio and cuenta_precio == 0:
                    categorias.append('Precio')
                    cuenta_precio = 1
                    
            return categorias

        sentimiento = sia.polarity_scores(reseña)
        categorias = calcular_categoria_sentimiento(reseña)

        if sentimiento['compound'] >= 0.05:
            st.write(f"Opinión positiva. Score: {sentimiento['compound']}")
        elif sentimiento['compound'] <= -0.05:
            st.write(f"Opinión negativa. Score: {sentimiento['compound']}")
        else:
            st.write(f"Opinión neutra. Score: {sentimiento['compound']}")

        st.write('La crítica trata los siguientes temas:')
        if len(categorias) == 0:
            st.write('General')
        else:
            for categoria in categorias:
                st.write(categoria)
    
