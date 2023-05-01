# Import para utilizar arquivos do drive no colab
#from google.colab import drive
#drive.mount('/content/drive')

#0 importar bibliotecas pandas, numpy, io
import pandas as pd
import numpy as np
import io



#1 importar o csv para um dataframe
df = pd.read_csv(#address, sep=',', na_values="?")



#2 verificar o número de linhas e colunas do dataframe
df.shape



#3 Utilizar o método info() para visualizar as informações do dataframe
df.info()



#4 head()
df.head()



#5 apagar a coluna id
#df.drop(['id'], axis = 1, inplace = True)
df = df.drop(['id'], axis = 1)

df.head()



#6 visualizar os campos com valores faltantes
df.isna().sum()



#7 apagar linhas duplicadas
df = df.drop_duplicates()



#8 número de linhas e colunas
df.shape



#9 visualizar os campos com valores faltantes
df.isna().sum()



#10 apagar linhas com todos os campos em branco
df = df.dropna(how = 'all', axis = 0)



#11 ver número de linhas e colunas
df.shape



#12 ver número de campos vazios nos atributos
df.isna().sum()



#13 preencher os campos vazios da coluna 'total_bedrooms' com o valor médio
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace = True)



#14 contar o número de campos faltantes
df.isna().sum()



#15 head()
df.head()



#18 como extrair uma amostra (sample) com n% do dataframe original
df.sample()



#18 como extrair uma amostra (sample) com n% do dataframe original
df.sample(frac=0.1)



#19 importar as bibliotecas matplotlib e seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



#20 construir um mini histograma de todos os atributos do dataframe
df.hist(bins=50, figsize=(20, 15))
plt.show()



#21 construir o histograma do atributo 'median_house_value' utilizando a biblioteca Seaborn.
sns.histplot(data = df, x = "median_house_value", kde = True, hue = "ocean_proximity", multiple = "stack")
plt.show()



#22 construir um gráfico boxplot do atributo 'median_house_value'
plt.boxplot(df['median_house_value'])
plt.title("Median House Value")
plt.show()



#23 construir um gráfico boxplot do atributo 'median_house_value' usando Seaborn
sns.boxplot(x = df["median_house_value"])
plt.title("Median House Value")
plt.show()



#24 construa um histograma do atributo 'households'
plt.hist(df['households'], bins = 50)
plt.title("Histograma HouseHolds")
plt.show()



#25 construa um histograma do atributo 'households' com seaborn
sns.histplot(data=df['households'], bins = 50, kde = True)
plt.show()



#26 Matriz de correlação 
corr = df.corr()
corr



#27 Visualização da Matriz de Correlação em um Mapa de calor
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot = False) #Annot = true mostra o valor da correlação
plt.show



#28 Vetor com o nome das colunas
df.columns



#29.1 Gráfico de dispersão feito direto com o pandas plot
df.plot(kind="scatter", # tipo de gráfico
        x="longitude",  # variável x
        y="latitude",   # variável y
        s=df["population"]/100, #o tamanho de cada ponto, neste caso, estamos relacionando ao tamanho da pop.
        label="population", # label
        figsize=(10,10),    # tamanho da figura
        c="median_house_value",# a cor de cada ponto, estamos mapeando de acordo com o valor 
        cmap=plt.get_cmap("jet"), #experimente trocar 'jet' por 'inferno'
        colorbar=True) #como mapeamos as cores pela coluna preço, a barra mostra a escala de preços
plt.show()



#29.2 apagar as colunas 'longitude' e 'latitude'
df = df.drop(['longitude', 'latitude'], axis = 1)



#29.3 utilize a função describe para visualizar um resumo estatísticos dos dados do seu dataframe
df.describe()



#30 Normalizar as escalas dos atributos do dataframe, retirar da normalização median_house_value (que desejamos fazer a predição )
# e ocean_proximity (porque é categórica)

from sklearn.preprocessing import Normalizer
escala = Normalizer()
df[['housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']]=escala.fit_transform(df[['housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']])



#31 head()
df.head()



#32 transformar a coluna 'ocean_proximity' in variável categorica
df.info()
# Forçar a mudança de tipo para categórica
df['ocean_proximity'] = df['ocean_proximity'].astype('category')
df.info()



#33 Contar quantas vezes cada categoria aparece na coluna ocean_proximity (Quantos tipos?)
df['ocean_proximity'].unique()



#34 Codificar o atributo 'ocean_proximity' com get_dummies
#Codifique o atributo categórico como uma matriz numérica one-hot.
df = pd.get_dummies(df, prefix = ['ocean_proximity'])



#35 head
df.sample(frac=0.10)



#36 Separar os atributos de entrada X e o atributo alvo de saída y
y = df['median_house_value']
x = df.drop(['median_house_value'], axis = 1)



#37 Dividindo o dataframe em dados de treino (80%) e dados de teste (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)



X_test.shape



#38 Aplicando o Modelo de Regressão Linear

# 1. importar a biblioteca LinearRegression
from sklearn.linear_model import LinearRegression

# 2. Estanciar o objeto do modelo a ser usado
model = LinearRegression()

# 3. Ajustar os dados ao modelo (fit)
model.fit(X_train, Y_train)

# 4. Testar o modelo com os dados de teste
Y_pred = model.predict(X_test)
Y_pred



#39 Mostrar os coeficientes do ajuste:
b = model.intercept_
b

a = model.coef_
a

# y = a1x1 + a2x2 + a3x3 ... a11x11



# 40. compara a predição com o esperado
from sklearn import metrics
from sklearn.metrics import r2_score

score = r2_score(Y_test, Y_pred)
score



#41 calculo das métricas MAE MSE RMSE
print('MAE - Mean Absolute Error:',     metrics.mean_absolute_error(Y_test, Y_pred))  
print('MSE - Mean Squared Error:',      metrics.mean_squared_error(Y_test, Y_pred))  
print('RMSE - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))  #mais usado