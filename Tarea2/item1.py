import numpy as np
import os
import sys
import random
from typing import List, Tuple
import time

def read_file(path):
    """
    - D : número de aviones (DOM).
    - E, P, L : arrays  de tiempo (temprano, prefetente, tardio).
    - Ci, Ck : arrays de costos por penalización por unidad bajo y sobre el prefetente.
    ..
    - tau : tiempo de separación minimos entre el aterrizaje minimo de dos aviones -> Tij.
    """
    with open(path, 'r') as file:
        data = file.readlines()

    index = 0
    num_planes = int(data[index].strip())
    index += 1

    E, P, L, Ci, Ck = [], [], [], [], []
    tau = []

    for _ in range(num_planes):
        values = data[index].strip().split()
        E.append(int(values[0]))
        P.append(int(values[1]))
        L.append(int(values[2]))
        Ci.append(float(values[3]))
        Ck.append(float(values[4]))
        index += 1

        sep_row = []
        while len(sep_row) < num_planes:
            sep_row.extend(list(map(int, data[index].strip().split())))
            index += 1
        tau.append(sep_row)

    E = np.array(E)
    P = np.array(P)
    L = np.array(L)
    Ci = np.array(Ci)
    Ck = np.array(Ck)
    tau = np.array(tau)

    return num_planes, E, P, L, Ci, Ck, tau

def calculate_cost(landing_times: np.ndarray, P: np.ndarray, Ci: np.ndarray, Ck: np.ndarray) -> float:
    """
    -> Se calcula el costo total de una solución de aterrizaje
    
    Parametros:
    - landing_times : tiempo asignados de aterrizaje para cada avion
    - P : tiempos preferentes de aterrizaje
    - Ci : costos por aterrizar antes del tiempo prefetente
    - Ck : costos por aterrizar despues del tiempo preferente
    
    Retorno -> Costo Total de la Solución
    """
    early_penalty = np.sum(Ci * np.maximum(0, P - landing_times))
    late_penalty = np.sum(Ck * np.maximum(0, landing_times - P))
    return early_penalty + late_penalty

def is_feasible(landing_times: List[int], plane_indices: List[int], tau: np.ndarray) -> bool:
    """
    Se encarga de verificar si un conjunto de tiempos de aterrizaje es factible (respeta las separaciones minimas).
    
    Parametros:
    - landing_times : lista de tiempos de aterrizaje asignados
    - plane_indices : indice de los aviones correspondientes a los tiempos de aterrizaje
    - tau : es la matriz de tiempos de separación minimos
    
    Retorno (boolean):
    - TRUE : si la solucion es FACTIBLE
    - FALSE : si la solucion es INFACTIBLE
    """
    for i in range(len(landing_times)):
        for j in range(i+1, len(landing_times)):
            i_idx, j_idx = plane_indices[i], plane_indices[j]
            if landing_times[i] + tau[i_idx][j_idx] > landing_times[j]:
                return False
            if landing_times[j] + tau[j_idx][i_idx] > landing_times[i]:
                return False
    return True

# -> ALGORITMOS GREEDY ...

def greedy_deterministic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray) -> Tuple[List[int], List[int], float]:
    """
    Algoritmo greedy determinista para la programación de aterrizajes, se encarga de:
    - ORDENAR LOS AVIONES POR TIEMPO PREFERENTE Y ASIGNARLOS UNO A UNO EN EL PRIMER TIEMPO FACTIBLE
    
    Parametros:
    - num_planes, E, P, L, Ci, Ck, tau: Son los datos del problema
    
    Retorno:
    - landing_times: lista de los tiempos de aterrizaje asignados
    - plane_order: orden en que se programaron los aviones
    - total_cost: costo total de la solución
    """
    # Ordenar aviones por tiempo preferente
    plane_indices = list(range(num_planes))
    plane_indices.sort(key=lambda i: P[i])
    
    landing_times = []
    assigned_planes = []
    
    for i in plane_indices:
        # Iniciar en el tiempo más temprano posible
        current_time = E[i]
        
        # Comprobar si este tiempo es factible con los aviones ya programados
        while True:
            temp_landing_times = landing_times + [current_time]
            temp_planes = assigned_planes + [i]
            
            if current_time <= L[i] and is_feasible(temp_landing_times, temp_planes, tau):
                landing_times.append(current_time)
                assigned_planes.append(i)
                break
            
            current_time += 1
            if current_time > L[i]:
                # Si llegamos aquí, no hay solución factible
                return None, None, float('inf')
    
    total_cost = calculate_cost(np.array(landing_times), P[assigned_planes], 
                              Ci[assigned_planes], Ck[assigned_planes])
    
    return landing_times, assigned_planes, total_cost

def greedy_stochastic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, seed: int = 42) -> Tuple[List[int], List[int], float]:
    """
    Algoritmo greedy estocastico para la programacion de aterrizajesm se espera:
    - SELECCIONAR ALEATORIAMENTE UN AVION DE LOS N MEJORES CANDIDATOS EN CADA ITERACION, BASANDOSE EN UNA HEURISTICA
    
    Parámetros:
    - num_planes, E, P, L, Ci, Ck, tau: son los datos del problema
    - seed: es la SEMILLA para el generador de números aleatorios
    
    Retorno:
    - landing_times: lista de tiempos de aterrizaje asignados
    - plane_order: orden en que se programaron los aviones
    - total_cost: costo total de la solución
    """
    random.seed(seed)
    
    # Lista de aviones disponibles (no asignados)
    available_planes = list(range(num_planes))
    
    landing_times = []
    assigned_planes = []
    
    # Continuar hasta que todos los aviones estén asignados
    while available_planes:
        # Calcular una heurística para cada avión disponible
        # Usaremos la suma ponderada de: tiempo preferente y costos de penalización
        candidates = []
        for i in available_planes:
            # Heurística: valor más bajo es mejor
            heuristic_value = P[i] + (Ci[i] + Ck[i]) / 2
            candidates.append((i, heuristic_value))
        
        # Ordenar candidatos por valor heurístico (menor es mejor)
        candidates.sort(key=lambda x: x[1])
        
        # Seleccionar aleatoriamente uno de los N mejores candidatos
        N = min(3, len(candidates))  # Podemos ajustar N según convenga
        selected_idx = random.randint(0, N-1)
        selected_plane = candidates[selected_idx][0]
        
        # Encontrar el primer tiempo factible para el avión seleccionado
        current_time = E[selected_plane]
        
        while True:
            temp_landing_times = landing_times + [current_time]
            temp_planes = assigned_planes + [selected_plane]
            
            if current_time <= L[selected_plane] and is_feasible(temp_landing_times, temp_planes, tau):
                landing_times.append(current_time)
                assigned_planes.append(selected_plane)
                available_planes.remove(selected_plane)
                break
            
            current_time += 1
            if current_time > L[selected_plane]:
                # Si llegamos aquí, no hay solución factible
                return None, None, float('inf')
    
    total_cost = calculate_cost(np.array(landing_times), P[assigned_planes], 
                              Ci[assigned_planes], Ck[assigned_planes])
    
    return landing_times, assigned_planes, total_cost

# -> SOLUCION FINAL ...

def print_solution(landing_times, plane_order, total_cost):
    """
    En este apartado se realiza la Impresion de la solucion de forma ordenada !!!
    """
    if landing_times is None:
        print("No se encontró solución factible.")
        return
    
    print(f"Costo total: {total_cost:.2f}")
    print("Avión\tTiempo de aterrizaje")
    print("-" * 20)
    
    for i, plane in enumerate(plane_order):
        print(f"{plane}\t{landing_times[i]}")

def run_multiple_stochastic(num_planes, E, P, L, Ci, Ck, tau, num_runs=10):
    """
    Ejecuta múltiples instancias del algoritmo greedy estocástico con diferentes semillas.
    """
    results = []
    
    for seed in range(num_runs):
        start_time = time.time()
        landing_times, plane_order, cost = greedy_stochastic(num_planes, E, P, L, Ci, Ck, tau, seed)
        end_time = time.time()
        
        results.append({
            "seed": seed,
            "cost": cost,
            "time": end_time - start_time,
            "landing_times": landing_times,
            "plane_order": plane_order
        })
    
    return results

def main():
    #filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case1.txt'
    
    select = int(input("Selecciona el caso a evaluar: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n"))
    
    if select == 0:
        sys.exit()
    
    # Ajustar las rutas según tu estructura de directorios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if select == 1:
        filename = os.path.join(script_dir, 'casos', 'case1.txt')
    elif select == 2:
        filename = os.path.join(script_dir, 'casos', 'case2.txt')
    elif select == 3:
        filename = os.path.join(script_dir, 'casos', 'case3.txt')
    elif select == 4:
        filename = os.path.join(script_dir, 'casos', 'case4.txt')
    else:
        print("Opción no válida")
        sys.exit()
    
    print(f"Buscando el archivo en: {filename}")

    try:
        num_planes, earliest, preferred, latest, early_cost, late_cost, separation = read_file(filename)
        print("-" * 40)
        print("Lectura del archivo exitosa.")
        print("-" * 40)
        
        print(f"Número de aviones (D): {num_planes}")
        print(f"Tiempos tempranos (E): {earliest}")
        print(f"Tiempos preferentes (P): {preferred}")
        print(f"Tiempos tardíos (L): {latest}")
        print(f"Costos penalización temprana (Ci/alpha): {early_cost}")
        print(f"Costos penalización tardía (Ck/beta): {late_cost}")
        print(f"Matriz de separación mínima (tau) [{separation.shape}]:")
        with np.printoptions(linewidth=np.inf):
            print(separation)
        
        print("\n" + "=" * 40)
        print("ALGORITMO GREEDY DETERMINISTA")
        print("=" * 40)
        
        start_time = time.time()
        landing_times, plane_order, cost = greedy_deterministic(num_planes, earliest, preferred, latest, early_cost, late_cost, separation)
        end_time = time.time()
        
        print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")
        print_solution(landing_times, plane_order, cost)
        
        print("\n" + "=" * 40)
        print("ALGORITMO GREEDY ESTOCÁSTICO (10 EJECUCIONES)")
        print("=" * 40)
        
        results = run_multiple_stochastic(num_planes, earliest, preferred, latest, early_cost, late_cost, separation, 10)
        
        # Mostrar resultados de las ejecuciones estocásticas
        print("\nResumen de resultados estocásticos:")
        print("Seed\tCosto\t\tTiempo (s)")
        print("-" * 40)
        
        for result in results:
            print(f"{result['seed']}\t{result['cost']:.2f}\t\t{result['time']:.4f}")
        
        # Encontrar la mejor solución
        best_result = min(results, key=lambda x: x['cost'])
        
        print("\nMejor solución estocástica:")
        print(f"Seed: {best_result['seed']}")
        print(f"Costo: {best_result['cost']:.2f}")
        print(f"Tiempo: {best_result['time']:.4f} segundos")
        print("\nDetalle de la mejor solución estocástica:")
        print_solution(best_result['landing_times'], best_result['plane_order'], best_result['cost'])
        
        # Comparación entre determinista y estocástico
        print("\n" + "=" * 40)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 40)
        print(f"Costo greedy determinista: {cost:.2f}")
        print(f"Costo mejor greedy estocástico: {best_result['cost']:.2f}")
        print(f"Diferencia: {cost - best_result['cost']:.2f}")
        print(f"Mejora porcentual: {((cost - best_result['cost']) / cost * 100) if cost != 0 else 0:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

if __name__ == '__main__':
    main()