import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Definición de la función de Himmelblau
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Función de recocido simulado
def simulated_annealing(func, bounds, temp, cooling_rate, stop_temp):
    # Genera una solución inicial aleatoria
    x = random.uniform(bounds[0], bounds[1])
    y = random.uniform(bounds[0], bounds[1])
    current_sol = np.array([x, y])
    current_energy = func(x, y)

    best_sol = np.copy(current_sol)
    best_energy = current_energy

    # Almacena los valores para visualizar el progreso
    history = []

    while temp > stop_temp:
        # Genera un nuevo candidato vecino de manera aleatoria
        new_sol = current_sol + np.random.uniform(-1, 1, size=2)
        new_sol = np.clip(new_sol, bounds[0], bounds[1])  # Se asegura de estar en el rango
        new_energy = func(new_sol[0], new_sol[1])

        # Calcula la diferencia de energía
        energy_diff = new_energy - current_energy

        # Decide si acepta la nueva solución
        if energy_diff < 0 or random.uniform(0, 1) < math.exp(-energy_diff / temp):
            current_sol = new_sol
            current_energy = new_energy

        # Actualiza la mejor solución encontrada
        if current_energy < best_energy:
            best_sol = np.copy(current_sol)
            best_energy = current_energy

        # Reduce la temperatura
        temp *= cooling_rate

        # Almacena los valores
        history.append((best_sol[0], best_sol[1], best_energy))

    return best_sol, best_energy, history

# Parámetros de la búsqueda
bounds = (-5, 5)  # Rango para x e y
initial_temp = 1000  # Temperatura inicial
cooling_rate = 0.99  # Tasa de enfriamiento
stop_temp = 1e-6  # Temperatura mínima de parada

# Ejecuta el algoritmo de recocido simulado
best_solution, best_energy, history = simulated_annealing(himmelblau, bounds, initial_temp, cooling_rate, stop_temp)

# Muestra los resultados
print("Mejor solución encontrada (x, y):", best_solution)
print("Valor mínimo de la función de Himmelblau:", best_energy)

# Visualización del progreso de la búsqueda
x_hist = [h[0] for h in history]
y_hist = [h[1] for h in history]
z_hist = [h[2] for h in history]

plt.plot(z_hist)
plt.title('Evolución de la función objetivo')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.show()
