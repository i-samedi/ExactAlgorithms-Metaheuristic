# Para cada caso se busca minimizar.
import random
import math
import numpy as np # Muy útil para operaciones vectoriales
import matplotlib.pyplot as plt

# --- Funciones Multimodales ---
def f1(x_vec):
    x1 = x_vec[0]
    return 4 - 4*x1**3 - 4*x1 + x1**2

def f2(x_vec):
    sum_terms = 0
    for i in range(6):
        xi = x_vec[i]
        sum_terms += (xi**2) * (2**(i+1)) 
    return (1/899) * (sum_terms - 1745)

def f3(x_vec):
    x1, x2 = x_vec[0], x_vec[1]
    return (x1**4 + x2**2 - 17)**2 + (2*x1 + x2 - 4)**2

def f4(x_vec): 
    sum_ln_terms = 0
    prod_x_terms = 1
    for i in range(10):
        xi = x_vec[i]
        # Asegurar que xi esté en (2, 10) para evitar errores de logaritmo o producto
        if not (2 < xi < 10): # Podrías manejar esto con penalizaciones o asegurarte en la generación
            return float('inf') # Mal fitness si está fuera de dominio
        sum_ln_terms += (math.log(xi - 2))**2 + (math.log(10 - xi))**2
        prod_x_terms *= xi
    return sum_ln_terms - (prod_x_terms**0.2)

# --- Parámetros de las funciones (dimensiones y límites) ---
# (min_bound, max_bound) para cada xi
FUNCTION_SPECS = {
    f1: {'dim': 1, 'bounds': [(-5, 5)]},
    f2: {'dim': 6, 'bounds': [(0, 1)] * 6},
    f3: {'dim': 2, 'bounds': [(-500, 500)] * 2},
    f4: {'dim': 10, 'bounds': [(2.00001, 9.99999)] * 10} # Dominio efectivo para f4
}

# --- 2. Componentes del Algoritmo Genético ---
def initialize_individual(dim, bounds):
    # bounds es una lista de tuplas [(min1, max1), (min2, max2), ...]
    individual = [random.uniform(b[0], b[1]) for b in bounds]
    return individual

def initialize_population(pop_size, dim, bounds):
    return [initialize_individual(dim, bounds) for _ in range(pop_size)]

def calculate_fitness(individual, obj_function):
    return obj_function(individual)

def tournament_selection(population, fitness_values, k=3):
    # Selecciona k individuos al azar, devuelve el mejor de ellos
    selected_indices = random.sample(range(len(population)), k)
    best_index_in_tournament = min(selected_indices, key=lambda i: fitness_values[i])
    return population[best_index_in_tournament]

def crossover_blx_alpha(parent1, parent2, alpha, bounds):
    child = []
    for i in range(len(parent1)):
        d = abs(parent1[i] - parent2[i])
        min_val = min(parent1[i], parent2[i]) - alpha * d
        max_val = max(parent1[i], parent2[i]) + alpha * d
        
        gene = random.uniform(min_val, max_val)
        
        # Aplicar límites (clamping)
        gene = max(bounds[i][0], min(gene, bounds[i][1]))
        child.append(gene)
    return child
    
def mutate_gaussian(individual, mutation_rate_per_gene, mutation_strength, bounds):
    mutated_individual = list(individual) # Copiar
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate_per_gene:
            change = random.gauss(0, mutation_strength * (bounds[i][1] - bounds[i][0])) # Escalar strength al rango
            mutated_individual[i] += change
            # Aplicar límites (clamping)
            mutated_individual[i] = max(bounds[i][0], min(mutated_individual[i], bounds[i][1]))
    return mutated_individual

# --- 3. Bucle Principal del GA ---
def genetic_algorithm(obj_function, specs, pop_size, num_generations, 
                      crossover_rate, mutation_rate_per_gene, mutation_strength,
                      tournament_k, elitism_count):
    
    dim = specs['dim']
    bounds = specs['bounds']
    
    population = initialize_population(pop_size, dim, bounds)
    best_fitness_history = []

    for generation in range(num_generations):
        fitness_values = [calculate_fitness(ind, obj_function) for ind in population]
        
        # Encontrar el mejor individuo de la generación actual
        current_best_idx = min(range(len(fitness_values)), key=fitness_values.__getitem__)
        current_best_individual = population[current_best_idx]
        current_best_fitness = fitness_values[current_best_idx]
        best_fitness_history.append(current_best_fitness)

        if generation % 10 == 0: # Imprimir progreso
            print(f"Gen: {generation}, Best Fitness: {current_best_fitness:.4f}")

        new_population = []

        # Elitismo
        if elitism_count > 0:
            sorted_indices = sorted(range(len(fitness_values)), key=fitness_values.__getitem__)
            for i in range(elitism_count):
                new_population.append(population[sorted_indices[i]])
        
        # Generar el resto de la nueva población
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values, tournament_k)
            parent2 = tournament_selection(population, fitness_values, tournament_k)
            
            child1 = parent1 # Por defecto
            if random.random() < crossover_rate:
                child1 = crossover_blx_alpha(parent1, parent2, 0.5, bounds) # alpha=0.5 es un valor común
            
            child1 = mutate_gaussian(child1, mutation_rate_per_gene, mutation_strength, bounds)
            new_population.append(child1)
            
            # Si necesitas dos hijos y tu operador de cruce los genera, ajusta aquí
            # if len(new_population) < pop_size and random.random() < crossover_rate:
            #    child2 = ...
            #    child2 = mutate_gaussian(child2, ...)
            #    new_population.append(child2)

        population = new_population[:pop_size] # Asegurar tamaño de población

    final_fitness_values = [calculate_fitness(ind, obj_function) for ind in population]
    best_idx = min(range(len(final_fitness_values)), key=final_fitness_values.__getitem__)
    
    return population[best_idx], final_fitness_values[best_idx], best_fitness_history


# --- 4. Experimentación ---
# Parámetros del GA (ESTOS SON LOS QUE DEBES VARIAR PARA LAS 4 CONFIGURACIONES)
CONFIGURATIONS = [
    {'pop_size': 50, 'num_generations': 100, 'crossover_rate': 0.8, 'mutation_rate_per_gene': 0.1, 'mutation_strength': 0.1, 'tournament_k': 3, 'elitism_count': 1},
    {'pop_size': 100, 'num_generations': 200, 'crossover_rate': 0.9, 'mutation_rate_per_gene': 0.05, 'mutation_strength': 0.05, 'tournament_k': 5, 'elitism_count': 2},
    # ... Añade dos configuraciones más ...
    # Justifica por qué eliges estos cambios (ej. más población para exploración, menos mutación para convergencia fina, etc.)
]

TARGET_FUNCTIONS = [f1, f2, f3, f4]

for func_obj in TARGET_FUNCTIONS:
    print(f"\n--- Optimizando {func_obj.__name__} ---")
    specs = FUNCTION_SPECS[func_obj]
    
    for i, config in enumerate(CONFIGURATIONS):
        print(f"\n  Configuración {i+1}: {config}")
        all_run_best_fitnesses = []
        all_run_histories = [] # Para promediar curvas de convergencia si quieres

        for run in range(10): # 10 ejecuciones
            print(f"    Run {run+1}/10")
            best_solution, best_fitness, history = genetic_algorithm(
                obj_function=func_obj,
                specs=specs,
                pop_size=config['pop_size'],
                num_generations=config['num_generations'],
                crossover_rate=config['crossover_rate'],
                mutation_rate_per_gene=config['mutation_rate_per_gene'],
                mutation_strength=config['mutation_strength'],
                tournament_k=config['tournament_k'],
                elitism_count=config['elitism_count']
            )
            all_run_best_fitnesses.append(best_fitness)
            all_run_histories.append(history)
            print(f"    Run {run+1} Best Fitness: {best_fitness:.6f}, Solution: {np.round(best_solution, 4)}")

        # Resultados para esta configuración y función
        avg_best_fitness = np.mean(all_run_best_fitnesses)
        std_best_fitness = np.std(all_run_best_fitnesses)
        overall_best_run_idx = np.argmin(all_run_best_fitnesses)
        
        print(f"\n  Resultados para Config {i+1} en {func_obj.__name__}:")
        print(f"    Mejor Fitness promedio en 10 runs: {avg_best_fitness:.6f} (std: {std_best_fitness:.6f})")
        print(f"    Mejor Fitness absoluto en 10 runs: {all_run_best_fitnesses[overall_best_run_idx]:.6f}")

        # Gráfico de convergencia (para una de las runs, o promediado)
        plt.figure()
        # Opción 1: Graficar la mejor run
        # plt.plot(all_run_histories[overall_best_run_idx], label=f"Mejor Run (Fitness: {all_run_best_fitnesses[overall_best_run_idx]:.4f})")
        
        # Opción 2: Graficar todas las runs
        for r_idx, hist in enumerate(all_run_histories):
             plt.plot(hist, alpha=0.3) # Curvas individuales semitransparentes
        
        # Opción 3: Graficar promedio y desviación estándar (más avanzado)
        # mean_history = np.mean(np.array(all_run_histories), axis=0)
        # std_history = np.std(np.array(all_run_histories), axis=0)
        # generations_axis = range(len(mean_history))
        # plt.plot(generations_axis, mean_history, label="Promedio Fitness")
        # plt.fill_between(generations_axis, mean_history - std_history, mean_history + std_history, alpha=0.2, label="Std Dev")

        # Usaremos la opción 1 o 2 para simplicidad aquí, para el informe, la 3 es más robusta.
        # Por ahora, graficaremos la historia de la primera run como ejemplo.
        plt.plot(all_run_histories[0], label=f"Run 1 (Fitness: {all_run_best_fitnesses[0]:.4f})")


        plt.title(f"Convergencia GA - {func_obj.__name__} - Config {i+1}")
        plt.xlabel("Generación")
        plt.ylabel("Mejor Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"convergence_{func_obj.__name__}_config{i+1}.png") # Guarda el gráfico
        plt.show() # Muestra el gráfico