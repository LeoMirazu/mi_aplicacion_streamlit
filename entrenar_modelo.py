import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Cargar los datos (reemplaza con tus datos reales o sintéticos)
data = pd.read_csv("C:\\Users\\Leonardo\\Modelos Refinería\\Datos sinteticos\\datos_petroleo_sinteticos.csv")

# 2. Dividir datos en características (X) y etiquetas (y)
X = data[['API', 'Azufre']]  # Entradas
y = data.iloc[:, 2:]  # Salidas (Curva de destilación)

# 3. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Guardar el modelo entrenado
joblib.dump(model, "modelo_random_forest.pkl")
print("Modelo entrenado y guardado como 'modelo_random_forest.pkl'")
