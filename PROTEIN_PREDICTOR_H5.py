# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import argparse
import h5py

def open_h5(h5_path):
    """
    Abre un archivo .h5 y lo convierte a dataframe.
    
    Args:
        h5_path (str): Ruta al archivo .h5 del embedding.
        
    Returns: 
        pd.DataFrame: DataFrame con los datos del embedding.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            nombres = []
            embeddings = []
            # Iterate over all datasets in the HDF5 file
            for name in hf:
                # 'name' is the key (protein identifier)
                # hf[name][()] is the data (the embedding)
                nombres.append(name)
                embeddings.append(hf[name][()])
            
        df = pd.DataFrame({
            'Nombre': nombres,
            'Embedding': embeddings
        })
        return df
    except Exception as e:
        print(f"Ocurrió un error al abrir el archivo .h5: {e}")
        return pd.DataFrame()

def predecir_svm(pesos_path, embeddings_path):
    """
    Predice usando un modelo SVM guardado. Se le pasa el archivo .plk del modelo y los embeddings a predecir.
    Produce una salida de texto de la clase de predicción y probabilidad de clase.
    
    Args:
        pesos_path (str): Ruta al archivo .plk donde está guardado el modelo.
        embeddings_path (str): Ruta al archivo con los embeddings a predecir.
        
    Returns:
        None: Imprime los resultados en formato texto.
    """
    # 1. Cargar el modelo
    try:
        model = joblib.load(pesos_path)
        print(f"Modelo cargado desde: {pesos_path}")
    except FileNotFoundError:
        print(f"Error: El archivo del modelo {pesos_path} no se encontró.")
        return
    except Exception as e:
        print(f"Ocurrió un error al cargar el modelo: {e}")
        return

    # 2. Cargar los embeddings desde el archivo .h5
    df_embeddings = open_h5(embeddings_path)
    if df_embeddings.empty:
        return
    
    # Convertir la columna 'Embedding' en un array 2D de NumPy para la predicción
    embeddings_array = np.vstack(df_embeddings['Embedding'].values)
    nombres = df_embeddings['Nombre'].tolist()

    # 3. Realizar las predicciones
    try:
        predicciones = model.predict(embeddings_array)
        probabilidades = model.predict_proba(embeddings_array)
        distancias = model.decision_function(embeddings_array)
    except Exception as e:
        print(f"Ocurrió un error durante la predicción: {e}")
        return

    # 4. Imprimir los resultados
    for i, nombre in enumerate(nombres):
        print(f"Nombre: {nombre}")
        print(f"Predicción: {predicciones[i]}")
        if predicciones[i] == 1:
            print("Clase: Alergeno")
        else:
            print("Clase: No Alergeno")
        print(f"Distancia al boundary: {round(distancias[i], 2)}")
        print(f"Probabilidad No Alergeno: {round(probabilidades[i][0], 2)}")
        print(f"Probabilidad Alergeno: {round(probabilidades[i][1], 2)}")
        print("-" * 30)

def main(pesos_path, embeddings_path):
    """
    Función principal para ejecutar el script de predicción.
    python PREDICTOR_PROTEINAS.py pesos_modelo.plk prueba.h5
    
    Args:
        pesos_path (str): Ruta al archivo .plk donde está guardado el modelo.
        embeddings_path (str): Ruta al archivo con los embeddings a predecir.
        
    Returns:
        None: Imprime los resultados en formato texto.
    """
    predecir_svm(pesos_path, embeddings_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para predecir usando un modelo SVM y embeddings desde un archivo .h5.")
    parser.add_argument("pesos_path", type=str, help="Ruta al archivo .plk donde está guardado el modelo.")
    parser.add_argument("embeddings_path", type=str, help="Ruta al archivo con los embeddings a predecir (formato .h5).")
    
    args = parser.parse_args()
    main(args.pesos_path, args.embeddings_path)