import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Cargar y procesar el archivo CSV (reemplaza 'tu_archivo.csv' con tu archivo real)
df_cleaned = pd.read_csv('data/datos.csv')
nombre_columna_carrera = 'Pregunta70'  # Ajusta esto según el nombre de tu columna
X = df_cleaned.drop(columns=[nombre_columna_carrera])
y = df_cleaned[nombre_columna_carrera]
y = pd.factorize(y)[0]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_categorical = to_categorical(y)

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Definir el modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')  # Salida para clasificación
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy del modelo: {accuracy * 100:.2f}%')

# Guardar el modelo y el scaler
model.save('applearning/modelo_vocacional_tf.h5')
joblib.dump(scaler, 'applearning/scaler.pkl')
