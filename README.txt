TP.02 : Clasificacion y validacion cruzada 
Nombre del grupo: JVC
Integrantes: Carlos Porcel, Valentino Pons, Ricardo Javier Suarez

 Para ejecutar correctamente el archivo 'sign_JVC.py' deben utilizar ciertas librerias
y en algunos casos es probable que deban instalar los modulos, adjuntamos tambien sus versiones.
Dejamos a continuacion lo que deben ejecutar.

Versiones de los modulos/librerias 
inline_sql==0.1.2
matplotlib==3.7.1
numpy==1.25.2
pandas==1.5.3
seaborn==0.13.1
sklearn==1.2.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier 
from inline_sql import sql, sql_val

 Para ejecutar el resto de los codigos que figuran en el archivo 'sign_JVC.py' pueden 
observar que estan agrupados por bloques (#%%). Para ello pueden ejecutar 
cada bloque con 'shift + enter'. 
