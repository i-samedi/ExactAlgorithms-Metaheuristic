import numpy as np
import os
import sys
import random
from typing import List, Tuple, Dict, Optional, Any
import time

def read_file(path):
    """
    - D : número de aviones (DOM).
    - E, P, L : arrays  de tiempo (temprano, prefetente, tardio).
    - Ci, Ck : arrays de costos por penalización por unidad bajo y sobre el prefetente.
    ..
    - tau : tiempo de separación minimos entre el aterrizaje minimo de dos aviones -> Tij.
    """
    try:
        with open(path, 'r') as file:
            data = file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        sys.exit(1)

    index = 0
    try:
        num_planes = int(data[index].strip())
        index += 1

        E, P, L, Ci, Ck = [], [], [], [], []
        tau_list = [] # Use a list of lists first

        for i in range(num_planes):
            values = data[index].strip().split()
            if len(values) < 5:
                raise ValueError(f"Error parsing plane {i+1} data line {index+1}: Expected 5 values, got {len(values)}")
            E.append(int(values[0]))
            P.append(int(values[1]))
            L.append(int(values[2]))
            Ci.append(float(values[3]))
            Ck.append(float(values[4]))
            index += 1

            # Read separation times for plane i
            sep_row = []
            while len(sep_row) < num_planes:
                if index >= len(data):
                     raise ValueError(f"Error parsing separation times for plane {i+1}: Unexpected end of file.")
                line_values = data[index].strip().split()
                if not line_values:
                    index += 1
                    continue
                try:
                    sep_row.extend(list(map(int, line_values)))
                except ValueError:
                     raise ValueError(f"Error parsing separation times on line {index+1}: Non-integer value found.")
                index += 1
            if len(sep_row) != num_planes:
                 raise ValueError(f"Error parsing separation times for plane {i+1}: Expected {num_planes} values, got {len(sep_row)}")
            tau_list.append(sep_row)

        if len(E) != num_planes or len(tau_list) != num_planes:
             raise ValueError("Mismatch between declared number of planes and data read.")
        for row in tau_list:
            if len(row) != num_planes:
                 raise ValueError("Separation matrix rows have inconsistent lengths.")

    except (ValueError, IndexError) as e:
        print(f"Error reading file format: {e}")
        sys.exit(1)

    E = np.array(E)
    P = np.array(P)
    L = np.array(L)
    Ci = np.array(Ci)
    Ck = np.array(Ck)
    tau = np.array(tau_list)

    if not np.all(E <= P) or not np.all(P <= L):
        print("Warning: Not all planes satisfy E <= P <= L constraint.")

    return num_planes, E, P, L, Ci, Ck, tau

def calculate_cost(schedule: Dict[int, int], P: np.ndarray, Ci: np.ndarray, Ck: np.ndarray) -> float:
    """
    Calculates the total cost of a given landing schedule dictionary.
    Uses numpy for potentially faster calculation if schedule is large.
    """
    if not schedule:
        return float('inf')

    plane_indices = np.array(list(schedule.keys()))
    landing_times = np.array(list(schedule.values()))

    # Ensure indices are valid before accessing P, Ci, Ck
    if np.any(plane_indices < 0) or np.any(plane_indices >= len(P)):
         print(f"Warning: Invalid plane indices found in schedule: {plane_indices[plane_indices >= len(P)]}")
         pass # Or return float('inf') ?

    P_scheduled = P[plane_indices]
    Ci_scheduled = Ci[plane_indices]
    Ck_scheduled = Ck[plane_indices]

    early_penalty = np.sum(Ci_scheduled * np.maximum(0, P_scheduled - landing_times))
    late_penalty = np.sum(Ck_scheduled * np.maximum(0, landing_times - P_scheduled))
    return early_penalty + late_penalty

# -> ALGORITMOS GREEDY CON SOPORTE PARA MÚLTIPLES PISTAS ...
Schedule = Dict[int, int]  # plane_idx -> landing_time
RunwayAssignment = Dict[int, int]  # plane_idx -> runway_idx

def greedy_deterministic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, 
                         Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, 
                         num_runways: int = 1) -> Tuple[Optional[Schedule], Optional[RunwayAssignment], float, bool]:
    """
    Algoritmo greedy determinista para la programación de aterrizajes, se encarga de:
    - ORDENAR LOS AVIONES POR TIEMPO PREFERENTE Y ASIGNARLOS UNO A UNO EN EL PRIMER TIEMPO FACTIBLE
    - SOPORTA MÚLTIPLES PISTAS DE ATERRIZAJE
    
    Parametros:
    - num_planes, E, P, L, Ci, Ck, tau: Son los datos del problema
    - num_runways: Número de pistas disponibles (1 o 2)
    
    Retorno:
    - schedule: Diccionario con los tiempos de aterrizaje asignados
    - runway_assignment: Diccionario con la pista asignada a cada avión
    - total_cost: Costo total de la solución
    - feasible: Indica si se encontró una solución factible
    """
    plane_indices = sorted(range(num_planes), key=lambda k: (P[k], E[k]))

    schedule: Schedule = {}
    runway_assignment: RunwayAssignment = {}
    
    # Para cada pista, mantener el último avión y su tiempo de aterrizaje
    last_plane_idx: List[int] = [-1] * num_runways
    last_landing_time: List[int] = [-1] * num_runways
    
    for k in plane_indices:
        # Para cada avión, encontrar la mejor pista y tiempo de aterrizaje
        best_landing_time = float('inf')
        best_runway = -1
        
        for runway in range(num_runways):
            min_start_time = E[k]
            earliest_after_separation = 0
            
            if last_plane_idx[runway] != -1:
                separation_needed = tau[last_plane_idx[runway]][k]
                if separation_needed >= 99999:
                    # Separación infactible para esta pista, intentar con otra
                    continue
                earliest_after_separation = last_landing_time[runway] + separation_needed
            
            actual_landing_time = max(min_start_time, earliest_after_separation)
            
            if actual_landing_time <= L[k] and actual_landing_time < best_landing_time:
                best_landing_time = actual_landing_time
                best_runway = runway
        
        if best_runway == -1:
            # No se encontró pista factible para este avión
            return None, None, float('inf'), False
        
        schedule[k] = best_landing_time
        runway_assignment[k] = best_runway
        last_plane_idx[best_runway] = k
        last_landing_time[best_runway] = best_landing_time
    
    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, runway_assignment, total_cost, True

def greedy_stochastic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, 
                       Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, 
                       num_runways: int = 1, rcl_size: int = 3, 
                       seed: Optional[int] = None) -> Tuple[Optional[Schedule], Optional[RunwayAssignment], float, bool]:   
    """
    Algoritmo greedy estocástico para la programación de aterrizajes, se espera:
    - SELECCIONAR ALEATORIAMENTE UN AVIÓN DE LOS N MEJORES CANDIDATOS EN CADA ITERACIÓN
    - SOPORTA MÚLTIPLES PISTAS DE ATERRIZAJE
    
    Parámetros:
    - num_planes, E, P, L, Ci, Ck, tau: son los datos del problema
    - num_runways: Número de pistas disponibles (1 o 2)
    - rcl_size: Tamaño de la lista restringida de candidatos
    - seed: es la SEMILLA para el generador de números aleatorios
    
    Retorno:
    - schedule: Diccionario con los tiempos de aterrizaje asignados
    - runway_assignment: Diccionario con la pista asignada a cada avión
    - total_cost: Costo total de la solución
    - feasible: Indica si se encontró una solución factible
    """
    if seed is not None:
        random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    schedule: Schedule = {}
    runway_assignment: RunwayAssignment = {}
    
    # Para cada pista, mantener el último avión y su tiempo de aterrizaje
    last_plane_idx: List[int] = [-1] * num_runways
    last_landing_time: List[int] = [-1] * num_runways

    while unscheduled_planes:
        candidates: List[Tuple[int, int, int]] = []  # (plane_idx, runway_idx, earliest_possible_time)

        for k in unscheduled_planes:
            for runway in range(num_runways):
                min_start_time = E[k]
                earliest_after_separation = 0
                
                if last_plane_idx[runway] != -1:
                    separation_needed = tau[last_plane_idx[runway]][k]
                    if separation_needed >= 99999:
                        continue
                    earliest_after_separation = last_landing_time[runway] + separation_needed
                
                earliest_possible_time = max(min_start_time, earliest_after_separation)
                
                if earliest_possible_time <= L[k]:
                    candidates.append((k, runway, earliest_possible_time))

        if not candidates:
            return None, None, float('inf'), False

        # Ordenar candidatos por tiempo más temprano de aterrizaje
        candidates.sort(key=lambda x: x[2])
        current_rcl_size = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_size]
        
        chosen_plane_idx, chosen_runway, chosen_landing_time = random.choice(rcl)

        schedule[chosen_plane_idx] = chosen_landing_time
        runway_assignment[chosen_plane_idx] = chosen_runway
        last_plane_idx[chosen_runway] = chosen_plane_idx
        last_landing_time[chosen_runway] = chosen_landing_time
        unscheduled_planes.remove(chosen_plane_idx)

    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, runway_assignment, total_cost, True

# -> UTILIDADES Y MAIN (Adaptado para mostrar información de pistas) ...
def print_solution(schedule: Optional[Schedule], runway_assignment: Optional[RunwayAssignment], 
                   total_cost: float, feasible: bool, num_runways: int):
    """
    Impresión de la solución de forma ordenada incluyendo asignación de pistas
    """
    if not feasible or schedule is None or runway_assignment is None:
        print("No se encontró solución factible.")
        return

    print(f"Costo total: {total_cost:.2f}")
    print("Avión\tPista\tTiempo de aterrizaje")
    print("-" * 35)

    # Ordenar primero por pista y luego por tiempo de aterrizaje
    sorted_schedule = []
    for plane_idx, landing_time in schedule.items():
        runway = runway_assignment[plane_idx]
        sorted_schedule.append((plane_idx, runway, landing_time))
    
    sorted_schedule.sort(key=lambda x: (x[1], x[2]))  # Ordenar por pista, luego por tiempo

    for plane_idx, runway, landing_time in sorted_schedule:
        # Usar índice base 1 para aviones y pistas para mayor claridad
        print(f"{plane_idx + 1}\t{runway + 1}\t{landing_time}")
    
    # Si hay más de una pista, mostrar estadísticas por pista
    if num_runways > 1:
        print("\nEstadísticas por pista:")
        for runway in range(num_runways):
            planes_in_runway = [p for p, r, _ in sorted_schedule if r == runway]
            print(f"Pista {runway + 1}: {len(planes_in_runway)} aviones")

def run_multiple_stochastic(num_planes, E, P, L, Ci, Ck, tau, num_runways=1, num_runs=10, rcl_size=3) -> List[Dict[str, Any]]:
    """
    Ejecuta múltiples instancias del algoritmo greedy estocástico.
    Returns list of result dictionaries.
    """
    results = []
    for i in range(num_runs):
        seed = i
        start_time = time.time()
        schedule, runway_assignment, cost, feasible = greedy_stochastic(
            num_planes, E, P, L, Ci, Ck, tau, 
            num_runways=num_runways, rcl_size=rcl_size, seed=seed
        )
        end_time = time.time()

        results.append({
            "seed": seed,
            "cost": cost,
            "time": end_time - start_time,
            "schedule": schedule,
            "runway_assignment": runway_assignment,
            "feasible": feasible
        })
    return results

def main():
    DEFAULT_CASE_DIR = "casos"

    select_str = input("Selecciona el caso a evaluar (1-4) o 0 para salir: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n")
    try:
        select = int(select_str)
        if select == 0:
            print("Saliendo...")
            sys.exit()
        if not 1 <= select <= 4:
            print("Opción no válida. Por favor, elija entre 1 y 4.")
            sys.exit()
    except ValueError:
        print("Entrada inválida. Por favor ingrese un número.")
        sys.exit()

    # Solicitar el número de pistas al usuario
    num_runways_str = input("\nSelecciona el número de pistas de aterrizaje (1-2): ")
    try:
        num_runways = int(num_runways_str)
        if not 1 <= num_runways <= 2:
            print("Número de pistas no válido. Se usará 1 pista por defecto.")
            num_runways = 1
    except ValueError:
        print("Entrada inválida. Se usará 1 pista por defecto.")
        num_runways = 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, DEFAULT_CASE_DIR, f'case{select}.txt')

    print(f"\nBuscando el archivo en: {filename}")
    print(f"Utilizando {num_runways} pista(s) de aterrizaje")

    try:
        num_planes, E, P, L, Ci, Ck, tau = read_file(filename)
        print("-" * 40)
        print("Lectura del archivo exitosa.")
        print(f"Número de aviones (D): {num_planes}")
        print(f"Tiempos tempranos (E): {E}")
        print(f"Tiempos preferentes (P): {P}")
        print(f"Tiempos tardíos (L): {L}")
        print(f"Costos penalización temprana (Ci/alpha): {Ci}")
        print(f"Costos penalización tardía (Ck/beta): {Ck}")
        print(f"Matriz de separación mínima (tau) [{tau.shape}]:")
        with np.printoptions(linewidth=np.inf):
            print(tau)
        print("-" * 40)

        # --- Greedy Determinista ---
        print("\n" + "=" * 40)
        print(f"ALGORITMO GREEDY DETERMINISTA ({num_runways} PISTA(S))")
        print("=" * 40)

        start_time_det = time.time()
        det_schedule, det_runway_assignment, det_cost, det_feasible = greedy_deterministic(
            num_planes, E, P, L, Ci, Ck, tau, num_runways=num_runways
        )
        end_time_det = time.time()

        print(f"Tiempo de ejecución: {end_time_det - start_time_det:.4f} segundos")
        print_solution(det_schedule, det_runway_assignment, det_cost, det_feasible, num_runways)

        # --- Greedy Estocástico ---
        print("\n" + "=" * 40)
        print(f"ALGORITMO GREEDY ESTOCÁSTICO ({num_runways} PISTA(S), 10 EJECUCIONES)")
        print("=" * 40)

        stochastic_results = run_multiple_stochastic(
            num_planes, E, P, L, Ci, Ck, tau, 
            num_runways=num_runways, num_runs=10, rcl_size=3
        )

        print("\nResumen de resultados estocásticos:")
        print("Seed\tCosto\t\tTiempo (s)")
        print("-" * 40)

        feasible_stochastic_results = []
        for result in stochastic_results:
            cost_str = f"{result['cost']:.2f}" if result['feasible'] else "inf"
            print(f"{result['seed']}\t{cost_str}\t\t{result['time']:.4f}")
            if result['feasible']:
                feasible_stochastic_results.append(result)

        # --- Find and Print Best Stochastic Solution ---
        if feasible_stochastic_results:
            best_result = min(feasible_stochastic_results, key=lambda x: x['cost'])

            print("\nMejor solución estocástica:")
            print(f"Seed: {best_result['seed']}")
            print(f"Costo: {best_result['cost']:.2f}")
            print(f"Tiempo: {best_result['time']:.4f} segundos")
            print("\nDetalle de la mejor solución estocástica:")
            print_solution(
                best_result['schedule'], 
                best_result['runway_assignment'], 
                best_result['cost'], 
                best_result['feasible'],
                num_runways
            )
        else:
            print("\nMejor solución estocástica:")
            print("No se encontraron soluciones factibles en las ejecuciones estocásticas.")
            best_result = None

        # --- Final Comparison ---
        print("\n" + "=" * 40)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 40)

        det_cost_display = f"{det_cost:.2f}" if det_feasible else "inf"
        best_sto_cost_display = f"{best_result['cost']:.2f}" if best_result else "inf"

        print(f"Costo greedy determinista: {det_cost_display}")
        print(f"Costo mejor greedy estocástico: {best_sto_cost_display}")

        if det_feasible and best_result:
            diff = det_cost - best_result['cost']
            if det_cost != 0:
                perc_improvement = (diff / det_cost * 100)
            else:
                perc_improvement = 0 if diff == 0 else float('inf')

            print(f"Diferencia: {diff:.2f}")
            print(f"Mejora porcentual: {perc_improvement:.2f}%")
        elif det_feasible and not best_result:
            print("Diferencia: N/A (Estocástico Infactible)")
            print("Mejora porcentual: N/A")
        elif not det_feasible and best_result:
            print("Diferencia: N/A (Determinista Infactible)")
            print("Mejora porcentual: N/A")
        else:
            print("Diferencia: N/A (Ambos Infactibles)")
            print("Mejora porcentual: N/A")

        print("=" * 40)

    except FileNotFoundError:
        print(f"Error Crítico: No se encontró el archivo de datos {filename}")
    except Exception as e:
        print(f"Error Crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()