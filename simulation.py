import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filters import moving_average

# Paramètres du capteur de pression (0 à 1 psi Sortie 14 bits de 10% à 90%)
P_MIN, P_MAX = 0.0, 1.0
COUNT_MIN, COUNT_MAX = 1638, 14746

def pressure_to_counts(pressure_psi):
    """
    Convertit la pression physique (psi) en fonction de la fonction de transfert du capteur.
    """
    numerator = (pressure_psi - P_MIN) * (COUNT_MAX - COUNT_MIN)
    denominator = P_MAX - P_MIN
    return (numerator / denominator) + COUNT_MIN

np.random.seed(42)
time = np.linspace(0, 2, 500) 

# Modélisation de l'impact du jet d'air par une impulsion gaussienne
theoretical_pressure = 0.6 * np.exp(-((time - 1.0) / 0.2)**2)
noise = np.random.normal(0, 0.03, size=len(time))
noisy_pressure = np.clip(theoretical_pressure + noise, 0.0, 1.0)

# Génération des données numériques purs, bruitées et filtrées
theoretical_counts = pressure_to_counts(theoretical_pressure)
noisy_counts = pressure_to_counts(noisy_pressure)
filtered_counts = moving_average(noisy_counts, window_size=7)

# Structuration des données
df = pd.DataFrame({
    'time': time,
    'theoretical_pressure_psi': theoretical_pressure,
    'theoretical_counts' : theoretical_counts,
    'noisy_counts': noisy_counts,
    'filtered_counts': filtered_counts
})

# Sauvegarde
os.makedirs('data_ia', exist_ok=True)
csv_path = 'data_ia/simulation_pressure.csv'
df.to_csv(csv_path, index=False)
print(f"Fichier '{csv_path}' généré avec succès.")

# Représentation graphique
plt.close('all')
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['noisy_counts'], label='Signal bruyant', color='red', alpha=0.5)
plt.plot(df['time'], df['filtered_counts'], label='Signal filtré', color='blue', linewidth=2)
plt.plot(df['time'], df['theoretical_counts'], label='Signal théorique pur', color='black', linestyle='--')

plt.title('Simulation et filtrage du capteur de pression')
plt.xlabel('Temps (en s)')
plt.ylabel('Sortie numérique (sur 14 bits)')
plt.legend()
plt.grid(True)
plt.show()