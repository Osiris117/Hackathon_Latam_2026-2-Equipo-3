from genetico_v3 import genetico
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")
capas1=30
capas2=15
accuracy=0.0
print("Ejecutando algoritmo genético...")
def entrenar_rna(capas1, capas2):
    rna = MLPClassifier(
        hidden_layer_sizes=(capas1, capas2),
        max_iter=1500,
        random_state=42
    )
    return rna
resultado = genetico()

df_hist = resultado["df_hist"]
mejor_secuencia = resultado["mejor_secuencia"]

df_hist["Decision_Optima_u"] = mejor_secuencia

print("\nEntrenando la Red Neuronal...")

X_train = df_hist[["Delta_S_obs", "S_actual"]]
y_train = df_hist["Decision_Optima_u"]

y_train_cat = y_train.astype(str)
rna=entrenar_rna(capas1, capas2)


rna.fit(X_train, y_train_cat)
# =========================================================
# EVALUACIÓN DEL MODELO
# =========================================================

# Predicciones sobre el conjunto de entrenamiento
y_pred = rna.predict(X_train)

# Exactitud
while accuracy < 95:  # Continuar hasta alcanzar el 95% de exactitud
    capas1 = random.randint(5, 30)
    capas2 = random.randint(3, 20)
    entrenar_rna(capas1, capas2)
    accuracy = accuracy_score(y_train_cat, y_pred)*100

print("\n==============================")
print(" RESULTADOS DEL ENTRENAMIENTO")
print("==============================")
print(f"Efectividad: {accuracy*100:.2f}%")
print(f"Aciertos   : {(y_pred == y_train_cat).sum()} de {len(y_train_cat)}")
print("==============================")
joblib.dump(rna, "modelo_rna_genetico.pkl")

print("\n¡Modelo guardado!")

plt.plot(rna.loss_curve_)
plt.title("Curva de Aprendizaje")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.grid(True)
plt.show()
