import numpy as np
import os
import sys
import random
import math
import time
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any, Set, Callable
from collections import defaultdict

# Se realiza la importación de las definiciones y funciones basicas del item 1
from item1 import read_file, calculate_cost, print_solution

# Definiciones de tipos
Schedule = Dict[int, int]  # plane_idx -> landing_time
RunwayAssignment = Dict[int, int]  # plane_idx -> runway_idx
Solution = Tuple[Schedule, RunwayAssignment]

class SimulatedAnnealing:
    """
    Implementación de Simulated Annealing para el problema de programación de aterrizajes.
    """
    
    def __init__(self, 
                 num_planes: int, 
                 E: np.ndarray, 
                 P: np.ndarray, 
                 L: np.ndarray, 
                 Ci: np.ndarray, 
                 Ck: np.ndarray, 
                 tau: np.ndarray,
                 num_runways: int = 1,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.95,
                 iterations: int = 1000,
                 neighbors_per_iter: int = 20,
                 reheating_threshold: Optional[int] = None,
                 reheating_factor: float = 1.5,
                 deterministic_start: bool = True,
                 name: str = "SA"):
        """
        Inicializa el algoritmo Simulated Annealing.
        
        Args:
            num_planes: Número de aviones a programar
            E, P, L: Tiempos temprano, preferente y tardío para cada avión
            Ci, Ck: Costos de penalización por aterrizar antes o después del tiempo preferente
            tau: Matriz de tiempos mínimos de separación entre aterrizajes
            num_runways: Número de pistas disponibles (1 o 2)
            initial_temp: Temperatura inicial
            cooling_rate: Tasa de enfriamiento (0 < cooling_rate < 1)
            iterations: Número máximo de iteraciones
            neighbors_per_iter: Número de vecinos a evaluar por iteración
            reheating_threshold: Número de iteraciones sin mejora antes de recalentar
            reheating_factor: Factor por el que se multiplica la temperatura al recalentar
            deterministic_start: Si es True, usa el greedy determinista como punto de partida
            name: Nombre para identificar esta configuración
        """
        self.num_planes = num_planes
        self.E = E
        self.P = P
        self.L = L
        self.Ci = Ci
        self.Ck = Ck
        self.tau = tau
        self.num_runways = num_runways
        
        # Parámetros del SA
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = iterations
        self.neighbors_per_iter = neighbors_per_iter
        self.reheating_threshold = reheating_threshold
        self.reheating_factor = reheating_factor
        self.deterministic_start = deterministic_start
        self.name = name
        
        # Variables para estadísticas
        self.iteration_costs = []
        self.best_costs = []
        self.temperatures = []
        self.accepted_moves = 0
        self.total_neighbors = 0
        self.reheating_events = 0
        
    def _is_feasible(self, schedule: Schedule, runway_assignment: RunwayAssignment) -> bool:
        """
        Verifica si la solución es factible respecto a todas las restricciones.
        
        Args:
            schedule: Diccionario de tiempos de aterrizaje
            runway_assignment: Diccionario de asignación de pistas
            
        Returns:
            bool: True si la solución es factible, False en caso contrario
        """
        if not schedule or not runway_assignment:
            return False
            
        # Verificar restricciones de ventana de tiempo para cada avión
        for plane, landing_time in schedule.items():
            if landing_time < self.E[plane] or landing_time > self.L[plane]:
                return False
                
        # Verificar restricciones de separación por pista
        runway_planes = defaultdict(list)
        for plane, runway in runway_assignment.items():
            runway_planes[runway].append(plane)
            
        # Para cada pista, verificar que los aviones respeten los tiempos de separación
        for runway, planes in runway_planes.items():
            # Ordenar aviones por tiempo de aterrizaje
            planes_sorted = sorted(planes, key=lambda p: schedule[p])
            
            # Verificar separación mínima entre aterrizajes consecutivos
            for i in range(len(planes_sorted) - 1):
                plane_i = planes_sorted[i]
                plane_j = planes_sorted[i + 1]
                if schedule[plane_j] < schedule[plane_i] + self.tau[plane_i][plane_j]:
                    return False
                    
        return True
        
    def _generate_neighbor(self, current_schedule: Schedule, current_runway_assignment: RunwayAssignment) -> Tuple[Schedule, RunwayAssignment]:
        """
        Genera una solución vecina aplicando uno de varios movimientos posibles.
        
        Args:
            current_schedule: Programación actual
            current_runway_assignment: Asignación de pistas actual
            
        Returns:
            new_schedule, new_runway_assignment: Nueva solución vecina
        """
        # Crear copias para no modificar los originales
        new_schedule = copy.deepcopy(current_schedule)
        new_runway_assignment = copy.deepcopy(current_runway_assignment)
        
        # Lista de posibles movimientos
        moves = [
            self._swap_two_planes,
            self._change_landing_time,
            self._change_runway
        ]
        
        # Seleccionar un movimiento al azar
        move = random.choice(moves)
        
        # Aplicar el movimiento hasta encontrar uno factible
        for _ in range(10):  # Intentar hasta 10 veces
            try:
                new_schedule, new_runway_assignment = move(new_schedule, new_runway_assignment)
                if self._is_feasible(new_schedule, new_runway_assignment):
                    return new_schedule, new_runway_assignment
            except:
                pass
                
        # Si no se pudo generar un vecino factible, devolver el original
        return current_schedule, current_runway_assignment
        
    def _swap_two_planes(self, schedule: Schedule, runway_assignment: RunwayAssignment) -> Tuple[Schedule, RunwayAssignment]:
        """
        Intercambia los tiempos de aterrizaje de dos aviones.
        """
        if len(schedule) < 2:
            return schedule, runway_assignment
            
        planes = list(schedule.keys())
        plane1, plane2 = random.sample(planes, 2)
        
        # Intercambiar tiempos de aterrizaje
        schedule[plane1], schedule[plane2] = schedule[plane2], schedule[plane1]
        
        return schedule, runway_assignment
        
    def _change_landing_time(self, schedule: Schedule, runway_assignment: RunwayAssignment) -> Tuple[Schedule, RunwayAssignment]:
        """
        Modifica el tiempo de aterrizaje de un avión aleatorio.
        """
        if not schedule:
            return schedule, runway_assignment
            
        plane = random.choice(list(schedule.keys()))
        
        # Determinar el rango factible para el nuevo tiempo
        min_time = self.E[plane]
        max_time = self.L[plane]
        
        # Generar un nuevo tiempo dentro del rango factible
        new_time = random.randint(min_time, max_time)
        schedule[plane] = new_time
        
        return schedule, runway_assignment
        
    def _change_runway(self, schedule: Schedule, runway_assignment: RunwayAssignment) -> Tuple[Schedule, RunwayAssignment]:
        """
        Cambia la asignación de pista de un avión aleatorio.
        Solo aplicable cuando hay más de una pista.
        """
        if self.num_runways <= 1 or not runway_assignment:
            return schedule, runway_assignment
            
        plane = random.choice(list(runway_assignment.keys()))
        
        # Cambiar a la otra pista
        runway_assignment[plane] = 1 - runway_assignment[plane]
        
        return schedule, runway_assignment
        
    def _get_initial_solution(self) -> Tuple[Schedule, RunwayAssignment, float, bool]:
        """
        Obtiene una solución inicial utilizando uno de los algoritmos greedy.
        
        Returns:
            schedule, runway_assignment, cost, feasible: Solución inicial y su costo
        """
        from item1 import greedy_deterministic, greedy_stochastic
        
        if self.deterministic_start:
            # Usar el greedy determinista
            return greedy_deterministic(
                self.num_planes, self.E, self.P, self.L, 
                self.Ci, self.Ck, self.tau, self.num_runways
            )
        else:
            # Usar el greedy estocástico con semilla aleatoria
            return greedy_stochastic(
                self.num_planes, self.E, self.P, self.L, 
                self.Ci, self.Ck, self.tau, self.num_runways,
                rcl_size=3, seed=random.randint(0, 100)
            )
            
    def run(self) -> Tuple[Schedule, RunwayAssignment, float, List[float], List[float]]:
        """
        Ejecuta el algoritmo Simulated Annealing.
        
        Returns:
            best_schedule, best_runway_assignment, best_cost, iteration_costs, temperatures
        """
        start_time = time.time()
        
        # Obtener solución inicial
        current_schedule, current_runway_assignment, current_cost, feasible = self._get_initial_solution()
        
        if not feasible:
            print(f"No se pudo encontrar una solución inicial factible. SA no puede continuar.")
            return None, None, float('inf'), [], []
            
        # Inicializar mejor solución
        best_schedule = copy.deepcopy(current_schedule)
        best_runway_assignment = copy.deepcopy(current_runway_assignment)
        best_cost = current_cost
        
        # Inicializar temperatura
        temp = self.initial_temp
        
        # Listas para seguimiento de resultados
        self.iteration_costs = [current_cost]
        self.best_costs = [best_cost]
        self.temperatures = [temp]
        
        # Contador para recalentamiento
        iterations_without_improvement = 0
        
        # Algoritmo principal de SA
        for iteration in range(self.max_iterations):
            # Explorar múltiples vecinos por iteración
            for _ in range(self.neighbors_per_iter):
                # Generar solución vecina
                new_schedule, new_runway_assignment = self._generate_neighbor(current_schedule, current_runway_assignment)
                self.total_neighbors += 1
                
                # Evaluar nuevo costo
                new_cost = calculate_cost(new_schedule, self.P, self.Ci, self.Ck)
                
                # Calcular diferencia de costo
                delta = new_cost - current_cost
                
                # Determinar si aceptar la nueva solución
                accept = False
                if delta <= 0:  # Mejora, aceptar siempre
                    accept = True
                else:  # Empeora, aceptar con cierta probabilidad
                    probability = math.exp(-delta / temp)
                    if random.random() < probability:
                        accept = True
                        
                # Actualizar solución actual si se acepta
                if accept:
                    current_schedule = new_schedule
                    current_runway_assignment = new_runway_assignment
                    current_cost = new_cost
                    self.accepted_moves += 1
                    
                    # Actualizar mejor solución si mejora
                    if current_cost < best_cost:
                        best_schedule = copy.deepcopy(current_schedule)
                        best_runway_assignment = copy.deepcopy(current_runway_assignment)
                        best_cost = current_cost
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                else:
                    iterations_without_improvement += 1
                    
            # Registrar costos para análisis
            self.iteration_costs.append(current_cost)
            self.best_costs.append(best_cost)
            
            # Enfriar la temperatura
            temp *= self.cooling_rate
            self.temperatures.append(temp)
            
            # Recalentar si es necesario
            if (self.reheating_threshold is not None and 
                iterations_without_improvement >= self.reheating_threshold):
                temp *= self.reheating_factor
                iterations_without_improvement = 0
                self.reheating_events += 1
                
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nSimulated Annealing ({self.name}) completado:")
        print(f"  Tiempo de ejecución: {execution_time:.4f} segundos")
        print(f"  Mejor costo: {best_cost:.2f}")
        print(f"  Movimientos aceptados: {self.accepted_moves}/{self.total_neighbors} ({self.accepted_moves/self.total_neighbors*100:.2f}%)")
        print(f"  Eventos de recalentamiento: {self.reheating_events}")
        
        return best_schedule, best_runway_assignment, best_cost, self.iteration_costs, self.temperatures

# 5 Configuraciones distintas de parametros para el algoritmo    
def run_multiple_configurations(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, num_runways: int = 1) -> List[Dict[str, Any]]:
    """
    Ejecuta múltiples configuraciones de Simulated Annealing.
    
    Returns:
        List[Dict]: Lista de resultados para cada configuración
    """
    # Definir configuraciones a probar
    configurations = [
        # Configuración 1: Enfriamiento lento, alta temperatura inicial
        {
            "initial_temp": 1000.0,
            "cooling_rate": 0.99,
            "iterations": 1000,
            "neighbors_per_iter": 10,
            "reheating_threshold": None,
            "reheating_factor": 1.0,
            "deterministic_start": True,
            "name": "Config 1 - Enfriamiento lento"
        },
        # Configuración 2: Enfriamiento rápido, alta temperatura
        {
            "initial_temp": 500.0,
            "cooling_rate": 0.90,
            "iterations": 1000,
            "neighbors_per_iter": 20,
            "reheating_threshold": None,
            "reheating_factor": 1.0,
            "deterministic_start": True,
            "name": "Config 2 - Enfriamiento rápido"
        },
        # Configuración 3: Con recalentamiento
        {
            "initial_temp": 200.0,
            "cooling_rate": 0.95,
            "iterations": 1000,
            "neighbors_per_iter": 15,
            "reheating_threshold": 100,
            "reheating_factor": 1.5,
            "deterministic_start": True,
            "name": "Config 3 - Con recalentamiento"
        },
        # Configuración 4: Más iteraciones, partida estocástica
        {
            "initial_temp": 200.0,
            "cooling_rate": 0.97,
            "iterations": 1500,
            "neighbors_per_iter": 10,
            "reheating_threshold": None,
            "reheating_factor": 1.0,
            "deterministic_start": False,
            "name": "Config 4 - Partida estocástica"
        },
        # Configuración 5: Temperatura baja, muchos vecinos
        {
            "initial_temp": 50.0,
            "cooling_rate": 0.95,
            "iterations": 800,
            "neighbors_per_iter": 30,
            "reheating_threshold": 150,
            "reheating_factor": 2.0,
            "deterministic_start": True,
            "name": "Config 5 - Muchos vecinos"
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"Ejecutando {config['name']}")
        print(f"{'='*50}")
        
        # Crear instancia de SA con la configuración actual
        sa = SimulatedAnnealing(
            num_planes, E, P, L, Ci, Ck, tau, num_runways,
            initial_temp=config["initial_temp"],
            cooling_rate=config["cooling_rate"],
            iterations=config["iterations"],
            neighbors_per_iter=config["neighbors_per_iter"],
            reheating_threshold=config["reheating_threshold"],
            reheating_factor=config["reheating_factor"],
            deterministic_start=config["deterministic_start"],
            name=config["name"]
        )
        
        # Ejecutar SA
        best_schedule, best_runway_assignment, best_cost, iteration_costs, temperatures = sa.run()
        
        # Guardar resultados
        results.append({
            "config": config,
            "best_schedule": best_schedule,
            "best_runway_assignment": best_runway_assignment,
            "best_cost": best_cost,
            "iteration_costs": iteration_costs,
            "temperatures": temperatures,
            "accepted_moves": sa.accepted_moves,
            "total_neighbors": sa.total_neighbors,
            "reheating_events": sa.reheating_events
        })
        
        # Imprimir detalles de la solución
        if best_schedule is not None:
            print_solution(best_schedule, best_runway_assignment, best_cost, True, num_runways)
        else:
            print("No se encontró solución factible.")
            
    return results

# Graficos comparativos de los resultados
def plot_results(results: List[Dict[str, Any]]) -> None:
    """
    Genera gráficos comparativos de los resultados de las diferentes configuraciones.
    """
    plt.figure(figsize=(15, 10))
    
    # Gráfico de evolución del costo
    plt.subplot(2, 1, 1)
    for result in results:
        if result["best_schedule"] is not None:
            config_name = result["config"]["name"]
            costs = result["iteration_costs"]
            plt.plot(costs, label=f"{config_name} (Mejor: {result['best_cost']:.2f})")
    
    plt.title("Evolución del Costo por Iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Costo")
    plt.legend()
    plt.grid(True)
    
    # Gráfico de evolución de la temperatura
    plt.subplot(2, 1, 2)
    for result in results:
        if result["best_schedule"] is not None:
            config_name = result["config"]["name"]
            temps = result["temperatures"]
            plt.plot(temps, label=config_name)
    
    plt.title("Evolución de la Temperatura")
    plt.xlabel("Iteración")
    plt.ylabel("Temperatura")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig("grafico_comparativo_item_3.png")
    print("\nGráfico guardado como 'grafico_comparativo_item_3.png'")
    
    # Tabla comparativa de resultados
    print("\nComparación de Configuraciones:")
    print(f"{'Configuración':<40} {'Mejor Costo':<15} {'Movimientos Aceptados':<25} {'Recalentamientos':<15}")
    print("-" * 95)
    
    for result in results:
        if result["best_schedule"] is not None:
            config_name = result["config"]["name"]
            best_cost = result["best_cost"]
            accepted = f"{result['accepted_moves']}/{result['total_neighbors']} ({result['accepted_moves']/result['total_neighbors']*100:.2f}%)"
            reheats = result["reheating_events"]
            
            print(f"{config_name:<40} {best_cost:<15.2f} {accepted:<25} {reheats:<15}")
        else:
            config_name = result["config"]["name"]
            print(f"{config_name:<40} {'INFACTIBLE':<15} {'-':<25} {'-':<15}")
            
def compare_with_greedy(num_planes: int, E: np.ndarray, P: np.ndarray, L: np.ndarray, Ci: np.ndarray, Ck: np.ndarray, tau: np.ndarray, num_runways: int, sa_results: List[Dict[str, Any]]) -> None:
    """
    Compara los resultados de SA con los algoritmos Greedy ORIGINALES DEL ITEM 1.
    """
    from item1 import greedy_deterministic, greedy_stochastic, run_multiple_stochastic
    
    print("\n" + "=" * 70)
    print("COMPARACIÓN CON ALGORITMOS GREEDY")
    print("=" * 70)
    
    # Ejecutar greedy determinista
    det_schedule, det_runway_assignment, det_cost, det_feasible = greedy_deterministic(
        num_planes, E, P, L, Ci, Ck, tau, num_runways
    )
    
    # Ejecutar mejores greedy estocásticos
    stochastic_results = run_multiple_stochastic(
        num_planes, E, P, L, Ci, Ck, tau, 
        num_runways=num_runways, num_runs=10, rcl_size=3
    )
    
    # Encontrar mejor resultado estocástico
    feasible_stochastic_results = [r for r in stochastic_results if r["feasible"]]
    if feasible_stochastic_results:
        best_sto_result = min(feasible_stochastic_results, key=lambda x: x["cost"])
        best_sto_cost = best_sto_result["cost"]
        best_sto_feasible = True
    else:
        best_sto_cost = float('inf')
        best_sto_feasible = False
    
    # Encontrar mejor resultado de SA
    feasible_sa_results = [r for r in sa_results if r["best_schedule"] is not None]
    if feasible_sa_results:
        best_sa_result = min(feasible_sa_results, key=lambda x: x["best_cost"])
        best_sa_cost = best_sa_result["best_cost"]
        best_sa_config = best_sa_result["config"]["name"]
    else:
        best_sa_cost = float('inf')
        best_sa_config = "N/A"
    
    # Mostrar tabla comparativa
    print("\nResumen de comparación:")
    print(f"{'Algoritmo':<40} {'Costo':<15} {'Mejora vs Det.':<15} {'Mejora vs Sto.':<15}")
    print("-" * 85)
    
    # Greedy determinista
    det_cost_str = f"{det_cost:.2f}" if det_feasible else "INFACTIBLE"
    print(f"{'Greedy Determinista':<40} {det_cost_str:<15} {'-':<15} {'-':<15}")
    
    # Mejor greedy estocástico
    sto_cost_str = f"{best_sto_cost:.2f}" if best_sto_feasible else "INFACTIBLE"
    if det_feasible and best_sto_feasible:
        sto_vs_det = f"{(det_cost - best_sto_cost) / det_cost * 100:.2f}%" if det_cost > 0 else "N/A"
    else:
        sto_vs_det = "N/A"
    print(f"{'Mejor Greedy Estocástico':<40} {sto_cost_str:<15} {sto_vs_det:<15} {'-':<15}")
    
    # Mejor Simulated Annealing
    sa_cost_str = f"{best_sa_cost:.2f}" if best_sa_cost < float('inf') else "INFACTIBLE"
    
    if det_feasible and best_sa_cost < float('inf'):
        sa_vs_det = f"{(det_cost - best_sa_cost) / det_cost * 100:.2f}%" if det_cost > 0 else "N/A"
    else:
        sa_vs_det = "N/A"
        
    if best_sto_feasible and best_sa_cost < float('inf'):
        sa_vs_sto = f"{(best_sto_cost - best_sa_cost) / best_sto_cost * 100:.2f}%" if best_sto_cost > 0 else "N/A"
    else:
        sa_vs_sto = "N/A"
        
    print(f"{'Mejor Simulated Annealing (' + best_sa_config + ')':<40} {sa_cost_str:<15} {sa_vs_det:<15} {sa_vs_sto:<15}")
    
    print("\nObservaciones:")
    if best_sa_cost < min(det_cost if det_feasible else float('inf'), 
                          best_sto_cost if best_sto_feasible else float('inf')):
        print("- Simulated Annealing logró mejorar los resultados de ambos algoritmos greedy.")
    elif best_sa_cost < (det_cost if det_feasible else float('inf')):
        print("- Simulated Annealing mejoró el resultado del greedy determinista pero no del estocástico.")
    elif best_sa_cost < (best_sto_cost if best_sto_feasible else float('inf')):
        print("- Simulated Annealing mejoró el resultado del greedy estocástico pero no del determinista.")
    else:
        print("- Simulated Annealing no logró mejorar los resultados de los algoritmos greedy.")

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
        print("-" * 40)
        
        # Ejecutar SA con múltiples configuraciones
        sa_results = run_multiple_configurations(
            num_planes, E, P, L, Ci, Ck, tau, num_runways
        )
        
        # Generar gráficos comparativos
        plot_results(sa_results)
        
        # Comparar con resultados de greedy
        compare_with_greedy(
            num_planes, E, P, L, Ci, Ck, tau, num_runways, sa_results
        )

    except FileNotFoundError:
        print(f"Error Crítico: No se encontró el archivo de datos {filename}")
    except Exception as e:
        print(f"Error Crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()