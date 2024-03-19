# -*- coding: utf-8 -*-
'''
Nombre de grupo: JVC
Integrantes: Carlos Porcel, Valentino Pons, Ricardo Javier Suarez

A continuacion se mostrará el codigo utilizado correspondiente al TP.02

El codigo esta dividido es la parte de la experimentacion y evaluacion de los modelos, 
y la otra parte esta relacionada al codigo usado para generar las imagenes del informe.

'''
#%%
#importo librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from inline_sql import sql , sql_val
from sklearn.tree import DecisionTreeClassifier

#%%
# import el dataframe
carpeta = ''
train2 = pd.read_csv(carpeta + "sign_mnist_train.csv")

#%%
# =============================================================================
#          CONSIGNA 1
# =============================================================================

'''
Acontinuacion se encuentra los experimentos relacinados con a responder:
¿La imagen corresponde a una seña de la L o a una seña de la A?

'''
# filtro train2 para quedarme con la A y L
df_a_l = train2.copy().loc[train2.label.isin([0,11])]

# Declaro mis variables
X = df_a_l.iloc[:,1:]
y = df_a_l[['label']]
k=2

# divido en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, shuffle=True, random_state=15, stratify= y )

#%%
# =============================================================================
#          FUNCIONES UTILIZADAS PARA LA SEGUNDA CONSIGNA
# =============================================================================

def knn_pixeles(atributos:[str],k:int ):
    
    '''
    Entrena y evalua un modelo de KNN con n_neighbors = 'k', usando los atributos dados en 'arbibutos'
    k: int
    atributos: lista de str

    Imprime Exactitud sobre datos de train, promedio de una validacion cruzada con k-folds = 5 y la exactitud sobre los datos de test. 

    Retorna el modelo entrenado.
    '''

    # modelo con pixeles del medio
    print('Entreno modelo con estos atributos '+ str(atributos) + ' y k = '+str(k))

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train[atributos],y_train.values.ravel())

    print('Exactitud con datos de train: %.2f' % knn.score(X_train[atributos],y_train))

    print('Validación cruzada con 5 k-folds: %.2f' % cross_val_score(knn, X_train[atributos], y_train.values.ravel(), cv=5).mean())

    print('Exactitud con datos de test: %.2f' % knn.score(X_test[atributos],y_test))

    return knn


def graf_r2_knn(atributos:[int]):
    '''
    Esta funcion grafica R2 en funcion de k (para train y test).
    k: desde 1 a 9
    atributos: es una lista de str con los tributos para entrenar el modelo
    '''

    # Rango de valores por los que se va a mover k
    valores_k = range(1, 10)
    #  Cantidad de veces que vamos a repetir el experimento
    Nrep = 100
    # Matrices donde vamos a ir guardando los resultados
    resultados_test  = np.zeros(( Nrep , len(valores_k)))
    resultados_train = np.zeros(( Nrep , len(valores_k)))

    # Realizamos la combinacion de todos los modelos (Nrep x k)
    for i in range(Nrep):
        # Dividimos en test(30%) y train(70%)
        X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X[atributos], y.values.ravel(), test_size = 0.3,shuffle=True, stratify=y,random_state=i)
        # Generamos el modelo y lo evaluamos
        for k in valores_k:
            # Declaramos el tipo de modelo
            neigh = KNeighborsClassifier(n_neighbors = k)
            # Entrenamos el modelo (con datos de train)
            neigh.fit(X_train_, Y_train_)
            # Evaluamos el modelo con datos de train y luego de test
            resultados_train[i,k-1] = neigh.score(X_train_, Y_train_)
            resultados_test[i,k-1]  = neigh.score(X_test_ , Y_test_ )

    # Promediamos los resultados de cada repeticion
    promedios_train = np.mean(resultados_train, axis = 0)
    promedios_test  = np.mean(resultados_test , axis = 0)

    ## Graficamos R2 en funcion de k (para train y test)
    plt.plot(valores_k, promedios_train, label = 'Train')
    plt.plot(valores_k, promedios_test, label = 'Test')
    plt.legend()
    plt.title('Performance del modelo de knn con '+str(len(atributos))+' atributos')
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('R^2')
    plt.xticks(valores_k)

#%%
# =============================================================================
#          FUNCIONES UTILIZADAS PARA LA TERCERA CONSIGNA
# =============================================================================

def precision(X_test, y_test , modelo):
    y_pred = modelo.predict(X_test)
    return (precision_score(y_test, y_pred, average='weighted'))

def recall(X_test, y_test , modelo):
    y_pred = arbol2.predict(X_test)
    return (recall_score(y_test, y_pred,average='weighted'))

def matriz_confusion_arbol(modelo_arb,datos,titulo):
    '''
    Grafica una matrix de confusion para el modelo_arb
    dado para los datos datos con el titulo que se le pasa
    '''
    y_pred = modelo_arb.predict(datos)

    matriz_confusion = confusion_matrix(y_test_v, y_pred)

    fig,ax = plt.subplots(figsize=(10, 8))

    ax = sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Etiquetas predichas' , fontsize= 16)
    ax.set_xticklabels(['A','E','I','O','U'],fontsize = 10)
    plt.ylabel('Etiquetas verdaderas' , fontsize= 16)
    ax.set_yticklabels(['A','E','I','O','U'],fontsize = 10)
    plt.title(titulo , fontsize=16)
    plt.show()
    
#%%
# =============================================================================
#          CONSIGNA 2
# =============================================================================

# Modelo entrenado con 3 pixeles del medio en forma horizontal

knn = knn_pixeles(['pixel392','pixel393','pixel394'], 2)
#%%
# Modelo entrenado con 3 pixeles del medio en forma vertical

knn_1 = knn_pixeles(['pixel365','pixel393','pixel421'],2)
#%%
# Modelo entrenado con 3 pixeles de la ultima fila en forma horizontal

knn_2 = knn_pixeles(['pixel769','pixel770','pixel771'],2)
#%%
#Modelo entrenado con 3 pixeles de la ultima fila en forma vertical
knn_2_ver = knn_pixeles(['pixel714','pixel742','pixel770'],2)


#%%
## Probamos con distintas cantidades de atributos tomando en cuenta el mejor modelo conseguido(pixeles de la muñeca)

# Modelo entrenado con 2 pixeles de la ultima fila en forma horizontal

knn_2_1 = knn_pixeles(['pixel769','pixel770'],2)
#%%
# Modelo entrenado con 4 pixeles de la ultima fila en forma horizontal

knn_2_2 = knn_pixeles(['pixel768','pixel769','pixel770','pixel771'],2)
#%%
# Modelo entrenado con 6 pixeles de la ultima fila en forma horizontal

knn_2_3 = knn_pixeles(['pixel767','pixel768','pixel769','pixel770','pixel771','pixel772'],2)

#%%
# =============================================================================
#          CONSIGNA 3
# =============================================================================

'''
Acontinuacion se encuentra los experimentos relacinados con a responder:
¿A cuál de las vocales corresponde la seña en la imagen?

'''
# creo mi dataframe con las vocales

vocales = ['0' ,'4' , '8' , '14' , '20']

solo_vocales = sql^"""
                SELECT *
                FROM train2
                WHERE label = '0' OR label = '4' OR label = '8'
                        OR label = '14' OR label = '20'
                """
# Declaro mis variables
X_v = solo_vocales.iloc[:,1:]
y_v = solo_vocales[['label']]
#%%
# divido en train y test
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y_v,test_size= 0.3, shuffle=True,random_state=15, stratify= y_v )

#%%
#planto mi arbol profundidad 2
arbol2 = DecisionTreeClassifier(criterion='entropy' , max_depth= 2)
arbol2.fit(X_train_v , y_train_v)
arbol2.score(X_test_v,y_test_v)
#%%
#arbol profundidad 4
arbol4 = DecisionTreeClassifier(criterion='entropy' , max_depth= 4)
arbol4.fit(X_train_v , y_train_v)
arbol4.score(X_test_v,y_test_v)
#%%
#arbol profundidad 8
arbol8 = DecisionTreeClassifier(criterion='entropy' , max_depth= 8)
arbol8.fit(X_train_v , y_train_v)
arbol8.score(X_test_v,y_test_v)
#%%
modelos_evaluados_en_train = pd.DataFrame({'metrica': ['Exactitud'],
                                  'arbol2' : [cross_val_score(arbol2, X_train_v, y_train_v, cv=5).mean()] ,

                                  'arbol4' : [cross_val_score(arbol4, X_train_v, y_train_v, cv=5).mean()],

                                  'arbol8' : [cross_val_score(arbol8, X_train_v, y_train_v, cv=5).mean()]
                                              })
#%%
modelos_evaluados_en_test = pd.DataFrame({'metrica': ['exactitud' , 'recall' , 'precision'],
                                  'arbol2' : [arbol2.score(X_test_v , y_test_v),recall(X_test_v , y_test_v , arbol2),
                                              precision(X_test_v , y_test_v , arbol2)],
                                  'arbol4' : [arbol4.score(X_test_v , y_test_v), recall(X_test_v , y_test_v , arbol4),
                                              precision(X_test_v , y_test_v , arbol4)],
                                  'arbol8' : [arbol8.score(X_test_v , y_test_v), recall(X_test_v , y_test_v , arbol8),
                                              precision(X_test_v , y_test_v , arbol8)],})

#%%
##########################################
##Codigo usado para generear las imagenes
#########################################


# genero un ejemplo de todas las letras en lengua de señas que estan en el conjunto de imagenes
fig , ax  = plt.subplots(4,6,figsize= (35,25))

label_sig = 0 

letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for i in range(4):
    for j in range(6):
        if label_sig == 9:
            label_sig+=1
        imgn = train2.loc[train2.label == label_sig].iloc[2].values
        ax[i,j].matshow(imgn[1:].reshape(28, 28), cmap = "gray")
        ax[i,j].set_title(letras[label_sig],fontsize=40)
        label_sig+=1
plt.show()



#%%
# b
#comparamos la letra E contra L y M
fig , ax  = plt.subplots(1,3,figsize= (15,15))

ax[0].matshow(train2.loc[train2.label == 4].iloc[0].values[1:].reshape(28, 28), cmap = "gray")
ax[0].set_title('Ejemplo letra E',fontsize=25)

ax[1].matshow(train2.loc[train2.label == 11].iloc[0].values[1:].reshape(28, 28), cmap = "gray")
ax[1].set_title('Ejemplo letra L',fontsize=25)

ax[2].matshow(train2.loc[train2.label == 12].iloc[0].values[1:].reshape(28, 28), cmap = "gray")
ax[2].set_title('Ejemplo letra M',fontsize=25)


plt.show()


#%%
# ejemplos 24 de lerta C
fig , ax  = plt.subplots(4,6,figsize= (35,25))

ind_ejem = 0 

for i in range(4):
    for j in range(6):
        if ind_ejem == 24:
            break
        imgn = train2.loc[train2.label == 2].iloc[ind_ejem].values
        ax[i,j].matshow(imgn[1:].reshape(28, 28), cmap = "gray")
        ax[i,j].set_title('Ejemplo: '+str(ind_ejem),fontsize=25)
        ind_ejem+=1
        
plt.show()

#%%
#Graficos punto 2

# grafico 
fig,ax = plt.subplots(figsize = (10,7))
ax = sns.countplot(data=df_a_l,x='label',stat='percent')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

ax.set_xticklabels(['A','L'],fontsize = 14)

#Agregar etiquetas y título al gráfico
plt.xlabel('Letras',fontsize=17)
plt.ylabel('Cantidad (%)',fontsize=17)
plt.title('Cantidad de A y L',fontsize=20)

#Mostrar el gráfico
plt.show()



#%%
# grafico de un ejemplo de las letras A y L
fig , ax  = plt.subplots(1,2,figsize= (10,5))
ejem = 5
ax[0].matshow(train2.loc[train2.label == 0].iloc[ejem].values[1:].reshape(28, 28), cmap = "gray")
ax[0].set_title('Ejemplo letra A',fontsize=25)

ax[1].matshow(train2.loc[train2.label == 11].iloc[ejem].values[1:].reshape(28, 28), cmap = "gray")
ax[1].set_title('Ejemplo letra L',fontsize=25)

plt.show()

#%%
# Grafico R^2 para KNN con k de 1 a 9, con 2 atributos de la munieca
graf_r2_knn(['pixel769','pixel770','pixel771'])
#%%
# Grafico R^2 para KNN con k de 1 a 9, con 3 atributos de la munieca
graf_r2_knn(['pixel769','pixel770'])
#%%
# Grafico R^2 para KNN con k de 1 a 9, con 6 atributos de la munieca
graf_r2_knn(['pixel767','pixel768','pixel769','pixel770','pixel771','pixel772'])

#%%
#genero matriz de confusion arbol2
matriz_confusion_arbol(arbol2, X_test_v,'Arbol2')

#%%
#genero matriz de confusion arbol4
matriz_confusion_arbol(arbol4, X_test_v,'Arbol4')

#%%
#genero matriz de confusion arbol8
matriz_confusion_arbol(arbol8, X_test_v,'Arbol8')
