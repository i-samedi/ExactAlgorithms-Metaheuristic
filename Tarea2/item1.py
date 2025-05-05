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

# -> ALGORITMOS GREEDY ...
Schedule = Dict[int, int]

def greedy_deterministic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray) -> Tuple[Optional[Schedule], float, bool]:
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
    plane_indices = sorted(range(num_planes), key=lambda k: (P[k], E[k]))

    schedule: Schedule = {}
    landing_sequence: List[int] = []
    last_plane_idx: int = -1
    last_landing_time: int = -1

    for k in plane_indices:
        min_start_time = E[k]
        earliest_after_separation = 0
        if last_plane_idx != -1:
            separation_needed = tau[last_plane_idx][k]
            if separation_needed >= 99999:
                 # print(f"Det: Infeasible separation tau[{last_plane_idx}][{k}]") # Verbose logging
                 return None, float('inf'), False
            earliest_after_separation = last_landing_time + separation_needed

        actual_landing_time = max(min_start_time, earliest_after_separation)

        if actual_landing_time > L[k]:
            # print(f"Det: Infeasible time for plane {k}. Required: {actual_landing_time}, Latest: {L[k]}. Sequence: {landing_sequence}") # Verbose logging
            return None, float('inf'), False

        schedule[k] = actual_landing_time
        landing_sequence.append(k)
        last_plane_idx = k
        last_landing_time = actual_landing_time

    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, total_cost, True

def greedy_stochastic(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, rcl_size: int = 3, seed: Optional[int] = None) -> Tuple[Optional[Schedule], float, bool]:   
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
    if seed is not None:
        random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    schedule: Schedule = {}
    landing_sequence: List[int] = []
    last_plane_idx: int = -1
    last_landing_time: int = -1

    while unscheduled_planes:
        candidates: List[Tuple[int, int]] = [] # (plane_idx, earliest_possible_time)

        for k in unscheduled_planes:
            min_start_time = E[k]
            earliest_after_separation = 0
            if last_plane_idx != -1:
                separation_needed = tau[last_plane_idx][k]
                if separation_needed >= 99999:
                    continue
                earliest_after_separation = last_landing_time + separation_needed

            earliest_possible_time = max(min_start_time, earliest_after_separation)

            if earliest_possible_time <= L[k]:
                candidates.append((k, earliest_possible_time))

        if not candidates:
            # print(f"Sto (seed={seed}): No feasible candidate. Unscheduled: {unscheduled_planes}. Last: {last_plane_idx}@{last_landing_time}. Seq: {landing_sequence}") # Verbose logging
            return None, float('inf'), False

        candidates.sort(key=lambda x: x[1]) # Sort by earliest possible time
        current_rcl_size = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_size]
        chosen_plane_idx, chosen_landing_time = random.choice(rcl)

        schedule[chosen_plane_idx] = chosen_landing_time
        landing_sequence.append(chosen_plane_idx)
        last_plane_idx = chosen_plane_idx
        last_landing_time = chosen_landing_time
        unscheduled_planes.remove(chosen_plane_idx)

    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, total_cost, True

# -> UTILIDADES Y MAIN (Adapted for Output Style) ...
def print_solution(schedule: Optional[Schedule], total_cost: float, feasible: bool):
    """
    En este apartado se realiza la Impresion de la solucion de forma ordenada !!!
    """
    # --- Match original output style ---
    if not feasible or schedule is None:
        print("No se encontró solución factible.") # Match original message
        # Optionally print cost inf if needed: print(f"Costo: {total_cost}")
        return

    # If feasible, print details
    print(f"Costo total: {total_cost:.2f}")
    print("Avión\tTiempo de aterrizaje") # Match original header
    print("-" * 25) # Match original separator length approx

    # Sort schedule by landing time for ordered printing
    sorted_schedule_items = sorted(schedule.items(), key=lambda item: item[1])

    for plane_idx, landing_time in sorted_schedule_items:
        # IMPORTANT: Print plane_idx + 1 to match 1-based indexing if that was the intention
        # in the original output. If original 'plane' meant the index, use plane_idx.
        # Assuming original 'plane' meant 1-based index:
        print(f"{plane_idx + 1}\t{landing_time}")
    # --- End match original output style ---

def run_multiple_stochastic(num_planes, E, P, L, Ci, Ck, tau, num_runs=10, rcl_size=3) -> List[Dict[str, Any]]:
    """
    Ejecuta múltiples instancias del algoritmo greedy estocástico.
    Returns list of result dictionaries.
    """
    results = []
    # No verbose printing here, summary will be printed later
    for i in range(num_runs):
        seed = i
        start_time = time.time()
        schedule, cost, feasible = greedy_stochastic(num_planes, E, P, L, Ci, Ck, tau, rcl_size=rcl_size, seed=seed)
        end_time = time.time()

        results.append({
            "seed": seed,
            "cost": cost,
            "time": end_time - start_time,
            "schedule": schedule, # Store the schedule dict
            "feasible": feasible
        })
    return results

def main():
    #filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case1.txt'
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, DEFAULT_CASE_DIR, f'case{select}.txt')

    print(f"\nBuscando el archivo en: {filename}") # Keep this line

    try:
        num_planes, E, P, L, Ci, Ck, tau = read_file(filename)
        print("-" * 40)
        print("Lectura del archivo exitosa.")
        # --- Keep original data print section ---
        print(f"Número de aviones (D): {num_planes}")
        print(f"Tiempos tempranos (E): {E}")
        print(f"Tiempos preferentes (P): {P}")
        print(f"Tiempos tardíos (L): {L}")
        print(f"Costos penalización temprana (Ci/alpha): {Ci}")
        print(f"Costos penalización tardía (Ck/beta): {Ck}")
        print(f"Matriz de separación mínima (tau) [{tau.shape}]:")
        with np.printoptions(linewidth=np.inf):
            print(tau)
        # --- End original data print section ---
        print("-" * 40)


        # --- Greedy Determinista ---
        print("\n" + "=" * 40)
        print("ALGORITMO GREEDY DETERMINISTA") # Match original header
        print("=" * 40)

        start_time_det = time.time()
        det_schedule, det_cost, det_feasible = greedy_deterministic(num_planes, E, P, L, Ci, Ck, tau)
        end_time_det = time.time()

        print(f"Tiempo de ejecución: {end_time_det - start_time_det:.4f} segundos") # Match original time print
        print_solution(det_schedule, det_cost, det_feasible) # Use adapted print function


        # --- Greedy Estocástico ---
        print("\n" + "=" * 40)
        print("ALGORITMO GREEDY ESTOCÁSTICO (10 EJECUCIONES)") # Match original header
        print("=" * 40)

        stochastic_results = run_multiple_stochastic(num_planes, E, P, L, Ci, Ck, tau, num_runs=10, rcl_size=3)

        # --- Display Stochastic Results Summary (Matching item1.py) ---
        print("\nResumen de resultados estocásticos:")
        print("Seed\tCosto\t\tTiempo (s)") # Match original table header
        print("-" * 40)

        feasible_stochastic_results = []
        for result in stochastic_results:
            cost_str = f"{result['cost']:.2f}" if result['feasible'] else "inf" # Format cost based on feasibility
            print(f"{result['seed']}\t{cost_str}\t\t{result['time']:.4f}")
            if result['feasible']:
                feasible_stochastic_results.append(result)
        # --- End stochastic results summary ---

        # --- Find and Print Best Stochastic Solution (Matching item1.py) ---
        if feasible_stochastic_results:
            best_result = min(feasible_stochastic_results, key=lambda x: x['cost'])

            print("\nMejor solución estocástica:") # Match original section header
            print(f"Seed: {best_result['seed']}")
            print(f"Costo: {best_result['cost']:.2f}")
            print(f"Tiempo: {best_result['time']:.4f} segundos")
            print("\nDetalle de la mejor solución estocástica:")
            # Call print_solution for the best result's schedule
            print_solution(best_result['schedule'], best_result['cost'], best_result['feasible'])
        else:
            # Handle case where no stochastic run was feasible
            print("\nMejor solución estocástica:")
            print("No se encontraron soluciones factibles en las ejecuciones estocásticas.")
            best_result = None # Ensure best_result is defined for comparison block
        # --- End best stochastic solution print ---


        # --- Final Comparison (Matching item1.py style) ---
        print("\n" + "=" * 40)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 40)

        # Use det_feasible flag and check best_result existence
        det_cost_display = f"{det_cost:.2f}" if det_feasible else "inf"
        best_sto_cost_display = f"{best_result['cost']:.2f}" if best_result else "inf"

        print(f"Costo greedy determinista: {det_cost_display}")
        print(f"Costo mejor greedy estocástico: {best_sto_cost_display}")

        # Calculate difference and percentage only if both are feasible
        if det_feasible and best_result:
            diff = det_cost - best_result['cost']
            # Handle division by zero or zero cost case for percentage
            if det_cost != 0:
                 perc_improvement = (diff / det_cost * 100)
            else:
                 perc_improvement = 0 if diff == 0 else float('inf') # Or handle as appropriate

            print(f"Diferencia: {diff:.2f}")
            print(f"Mejora porcentual: {perc_improvement:.2f}%")
        elif det_feasible and not best_result:
             print("Diferencia: N/A (Estocástico Infactible)")
             print("Mejora porcentual: N/A")
        elif not det_feasible and best_result:
             print("Diferencia: N/A (Determinista Infactible)")
             print("Mejora porcentual: N/A")
        else: # Both infeasible
             print("Diferencia: N/A (Ambos Infactibles)")
             print("Mejora porcentual: N/A")

        print("=" * 40)
        # --- End final comparison ---

    except FileNotFoundError:
        print(f"Error Crítico: No se encontró el archivo de datos {filename}")
    except Exception as e:
        print(f"Error Crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()