import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = joblib.load("modelo_random_forest.pkl")

# Ingresar nuevos datos para predicción
# Cambia estos valores por los que desees predecir
nuevo_petroleo = pd.DataFrame({'API': [32.11], 'Azufre': [6000]})

# Hacer predicción
curva_predicha = model.predict(nuevo_petroleo)

# Mostrar los resultados de la predicción
print("Curva de Destilación Predicha (Temperaturas en °C):")
print(curva_predicha)

# Visualizar la curva predicha
volumenes = np.linspace(0, 100, curva_predicha.shape[1])  # Porcentajes de volumen destilado
plt.plot(volumenes, curva_predicha[0], label="Curva Predicha")
plt.title("Curva de Destilación Predicha")
plt.xlabel("Porcentaje de Volumen Destilado (%)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.show()
