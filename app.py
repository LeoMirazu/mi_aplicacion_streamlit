import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = joblib.load("modelo_random_forest.pkl")

# Configuración de la aplicación
st.title("Predicción de Curva de Destilación ASTM D86")
st.markdown("Esta aplicación permite predecir la curva de destilación para un petróleo a partir de su gravedad API y contenido de azufre.")

# Entrada del usuario
api = st.number_input("Gravedad API", min_value=20.0, max_value=45.0, value=32.11)
azufre = st.number_input("Contenido de Azufre (ppm)", min_value=500, max_value=10000, value=6000)

# Botón para predecir
if st.button("Predecir"):
    # Crear DataFrame con los datos ingresados
    nuevo_petroleo = pd.DataFrame({'API': [api], 'Azufre': [azufre]})

    # Hacer predicción
    curva_predicha = model.predict(nuevo_petroleo)[0]

    # Crear tabla de resultados con los volúmenes especificados
    volumenes = np.linspace(0, 100, len(curva_predicha))  # % de volumen destilado
    vol_destilados = np.arange(0, 100, 10)  # Volúmenes de interés: 0%, 10%, ..., 90%
    temperaturas_interpoladas = np.interp(vol_destilados, volumenes, curva_predicha)

    resultados = pd.DataFrame({
        "Volumen Destilado (%)": vol_destilados,
        "Temperatura (°C)": temperaturas_interpoladas
    })

    # Mostrar resultados
    st.subheader("Curva de Destilación Predicha")
    st.line_chart(pd.DataFrame({"Temperatura (°C)": curva_predicha}, index=volumenes))

    st.subheader("Tabla de Resultados")
    st.dataframe(resultados)

    # Mostrar gráfico adicional para los volúmenes de interés
    fig, ax = plt.subplots()
    ax.plot(volumenes, curva_predicha, label="Curva Completa")
    ax.scatter(vol_destilados, temperaturas_interpoladas, color="red", label="Puntos Interpolados")
    ax.set_title("Curva de Destilación Predicha")
    ax.set_xlabel("Porcentaje de Volumen Destilado (%)")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend()
    st.pyplot(fig)
