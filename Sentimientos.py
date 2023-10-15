#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
from translate import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from nltk.corpus import stopwords
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


# In[5]:


nltk.download('vader_lexicon')
nltk.download('punkt')


# In[12]:


tables_info=[]
for i in range (1,14):
    url= "https://www.booking.com/reviews/es/hotel/flamero.es.html?label=gen173nr-1FCA0oRkIHZmxhbWVyb0gKWARoRogBAZgBCrgBF8gBDNgBAegBAfgBAogCAagCA7gCrfCeqQbAAgHSAiRhODg5OWY3NC00MzE4LTQ1ZjQtOGI5Mi0yZGFlYmRmZWM0NjTYAgXgAgE&sid=d9ddf77f00ff19eea1916cb4305df56a&customer_type=total&hp_nav=0&old_page=0&order=featuredreviews&page="+str(i)+"&r_lang=es&rows=75&"
    headers = {"User-Agent":"Mozilla/5.0"}
    req = Request(url,headers=headers)
    raw_web = urlopen(req).read()
    soup = BeautifulSoup(raw_web, 'html.parser')
    tables_info.append(soup.find_all('span', attrs={"itemprop": "reviewBody"}))


# In[13]:


tables_info


# In[18]:


reseñas=[]
for fil in range(0,13): 
    for j in tables_info[fil]:
        reseñas.append(j.text.strip())
reseñas


# In[19]:


sia = SentimentIntensityAnalyzer()


# In[24]:


frase_ingles = []
for i in reseñas:
    translator = Translator(from_lang="es", to_lang="en")
    traduccion = translator.translate(i)
    frase_ingles.append(traduccion)
frase_ingles


# In[25]:


palabras_positivas = ["good","happy","big","recommend","nice"]
palabras_negativas = ["old","small","uncomfortable","bad","slow"]


# In[26]:


def calcular_puntuacion_sentimiento(frase_ingles):
    tokens = nltk.word_tokenize(frase_ingles)
    puntuacion_sentimiento = 0

    for token in tokens:
        if token in palabras_positivas:
            puntuacion_sentimiento += 1
        elif token in palabras_negativas:
            puntuacion_sentimiento -= 1

    return puntuacion_sentimiento


# In[27]:


opiniones_positivas=0
opiniones_negativas=0
opiniones_neutras=0
for j in frase_ingles:
    puntuacion = calcular_puntuacion_sentimiento(j)
    if puntuacion > 0:
        opiniones_positivas=opiniones_positivas+1
    elif puntuacion < 0:
        opiniones_negativas=opiniones_negativas+1
    else:
        sentimiento = sia.polarity_scores(j)

        if sentimiento['compound'] >= 0.05:
            opiniones_positivas=opiniones_positivas+1
        elif sentimiento['compound'] <= -0.05:
            opiniones_negativas=opiniones_negativas+1
        else:
            opiniones_neutras=opiniones_neutras+1
print("Hay",opiniones_positivas,"opiniones positivas")
print("Hay",opiniones_negativas,"opiniones negativas")
print("Hay",opiniones_neutras,"opiniones neutras")

