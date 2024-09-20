import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

#Estou usando apenas a granulometria no treinamento e previsão de novos valores de fc (Essa rede usa retropropagação -  (model.fit)

# Função para normalizar os dados
def normalizar(dados):

    # Separar a primeira linha e as demais linhas
    primeira_linha = dados.iloc[0]
    dados_sem_primeira_linha = dados.iloc[1:]
    
    scaler_fc = MinMaxScaler()
    scaler_dim_max = MinMaxScaler()

    # Normalizando as colunas separadamente
    dados_normalizados = dados_sem_primeira_linha.copy()
    dados_normalizados[['fc']] = scaler_fc.fit_transform(dados_sem_primeira_linha[['fc']])
    dados_normalizados[['dim_max']] = scaler_dim_max.fit_transform(dados_sem_primeira_linha[['dim_max']])

    # Reincluir a primeira linha no DataFrame normalizado
    dados_normalizados = pd.concat([pd.DataFrame([primeira_linha]), dados_normalizados]).reset_index(drop=True)

    return dados_normalizados, scaler_fc, scaler_dim_max

# Carregando os dados
dados = pd.read_csv("C:/Users/bianc/OneDrive/Área de Trabalho/ic/Banco de dados/Rede Neural/dados-novo.csv")

# Normalização dos dados
dados_normalizados, scaler_fc, scaler_dim_max = normalizar(dados)

# Separação das variáveis (dependentes e independentes)
x = dados_normalizados[['fc']]  # Variável independente
y = dados_normalizados['dim_max']  # Variável dependente

# Separação das amostras (30% para teste, 70% para treino)
x_train, x_test, y_train, y_test = split(x, y, test_size=0.3, random_state=42)

# Definição do modelo
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))  # Testei reduzir o número de neurônios
model.add(Dropout(0.1))  # Adicionei dropout para tentar evitar overfitting (desativação)
model.add(Dense(32, activation='relu'))  # Segunda camada com menos neurônios
model.add(Dense(1, activation='linear')) 


optimizer = Adam(learning_rate=0.0001)

# Compilação do modelo com regularização e métrica de erro absoluto
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Callback para reduzir o learning rate se o modelo parar de melhorar
# Reduz a taxa de aprendizado em 20% (factor) se a perda de validação (val_loss) não melhorar por 5 épocas (patience)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Treinamento
history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.3, callbacks=[reduce_lr])

# Avaliação
loss, mae = model.evaluate(x_train, y_train)
print(f'Mean Absolute Error (MAE): {mae}')

# Fazer previsões
dim_max_novos = model.predict(x_test)

# Desnormalizar as previsões para a escala original
dim_sem_normalizar = scaler_dim_max.inverse_transform(dim_max_novos)

print("Predições do diâmetro máximo do agregado (escala original)")
print(dim_sem_normalizar)

# Plotar loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.title('Curva de Aprendizado (Loss)')
plt.show()

# Plotar MAE
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Épocas')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()
plt.title('Curva de Aprendizado (MAE)')
plt.show()
