import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Funciones Multimodales presentadas en la Tarea ---
def f1(x_vec):
    x1 = x_vec[0]
    if not (-5 <= x1 <= 5): return float('inf')
    return 4 - 4*x1**3 - 4*x1 + x1**2

def f2(x_vec):
    sum_terms = 0
    for i in range(6):
        xi = x_vec[i]
        if not (0 <= xi <= 1): return float('inf')
        sum_terms += (xi**2) * (2**(i+1))
    return (1/899) * (sum_terms - 1745)

def f3(x_vec):
    x1, x2 = x_vec[0], x_vec[1]
    if not ((-500 <= x1 <= 500) and (-500 <= x2 <= 500)): return float('inf')
    return (x1**4 + x2**2 - 17)**2 + (2*x1 + x2 - 4)**2

def f4(x_vec):
    sum_ln_terms = 0
    prod_x_terms = 1
    for i in range(10):
        xi = x_vec[i]
        if not (2.00001 < xi < 9.99999):
            return float('inf')
        sum_ln_terms += (math.log(xi - 2))**2 + (math.log(10 - xi))**2
        prod_x_terms *= xi
    return sum_ln_terms - (prod_x_terms**0.2)

# --- Parámetros de las funciones (dimensiones y límites) ---
FUNCTION_SPECS = {
    f1: {'dim': 1, 'bounds': [(-5, 5)]},
    f2: {'dim': 6, 'bounds': [(0, 1)] * 6},
    f3: {'dim': 2, 'bounds': [(-500, 500)] * 2},
    f4: {'dim': 10, 'bounds': [(2.00001, 9.99999)] * 10}
}

# --- 2. Componentes del Algoritmo Genético (GA) ---
def initialize_individual(dim, bounds):
    return [random.uniform(b[0], b[1]) for b in bounds]

def initialize_population(pop_size, dim, bounds): 
    return [initialize_individual(dim, bounds) for _ in range(pop_size)]

def calculate_fitness(individual, obj_function):
    return obj_function(individual)

def tournament_selection(population, fitness_values, k=3):
    selected_indices = random.sample(range(len(population)), k)
    # Manejar caso donde todos los fitness son inf
    valid_indices = [idx for idx in selected_indices if fitness_values[idx] != float('inf')]
    if not valid_indices: # Si todos son inf, elige uno al azar de los seleccionados
        return population[random.choice(selected_indices)]
    
    best_index_in_tournament = min(valid_indices, key=lambda i: fitness_values[i])
    return population[best_index_in_tournament]

def crossover_blx_alpha(parent1, parent2, alpha, bounds):
    child = []
    for i in range(len(parent1)):
        d = abs(parent1[i] - parent2[i])
        min_val_gene = min(parent1[i], parent2[i]) - alpha * d
        max_val_gene = max(parent1[i], parent2[i]) + alpha * d
        gene = random.uniform(min_val_gene, max_val_gene)
        gene = max(bounds[i][0], min(gene, bounds[i][1])) # Clamping
        child.append(gene)
    return child
    
def mutate_gaussian(individual, mutation_rate_per_gene, mutation_strength_factor, bounds):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate_per_gene:
            gene_range = bounds[i][1] - bounds[i][0]
            sigma = max(mutation_strength_factor * gene_range, 1e-6) 
            change = random.gauss(0, sigma)
            mutated_individual[i] += change
            mutated_individual[i] = max(bounds[i][0], min(mutated_individual[i], bounds[i][1])) # Clamping
    return mutated_individual

# --- 3. Bucle Principal del GA ---
def genetic_algorithm(obj_function, specs, pop_size, num_generations, 
                      crossover_rate, mutation_rate_per_gene, mutation_strength_factor,
                      tournament_k, elitism_count):
    
    dim = specs['dim']
    bounds = specs['bounds']
    
    population = initialize_population(pop_size, dim, bounds)
    best_fitness_history = []
    best_overall_solution = None
    best_overall_fitness = float('inf')

    for generation in range(num_generations):
        fitness_values = [calculate_fitness(ind, obj_function) for ind in population]
        
        current_gen_best_fitness = float('inf')
        current_gen_best_individual = None

        for i in range(len(population)):
            if fitness_values[i] < current_gen_best_fitness:
                current_gen_best_fitness = fitness_values[i]
                current_gen_best_individual = population[i]
        
        if current_gen_best_fitness < best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_solution = list(current_gen_best_individual) 

        best_fitness_history.append(best_overall_fitness)

        if generation % (num_generations // 10) == 0 or generation == num_generations -1 :
             print(f"  Gen: {generation:3d}, Best Fitness: {best_overall_fitness:.6f}")

        new_population = []

        if elitism_count > 0:
            valid_pop_fitness = [(pop, fit) for pop, fit in zip(population, fitness_values) if fit != float('inf')]
            if valid_pop_fitness:
                sorted_pop = sorted(valid_pop_fitness, key=lambda x: x[1])
                for i in range(min(elitism_count, len(sorted_pop))):
                    new_population.append(sorted_pop[i][0])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values, tournament_k)
            parent2 = tournament_selection(population, fitness_values, tournament_k)
            
            child = parent1
            if random.random() < crossover_rate:
                child = crossover_blx_alpha(parent1, parent2, 0.5, bounds) 
            
            child = mutate_gaussian(child, mutation_rate_per_gene, mutation_strength_factor, bounds)
            new_population.append(child)
            
        population = new_population[:pop_size]

    final_fitness_values = [calculate_fitness(ind, obj_function) for ind in population]
    final_best_idx = 0
    final_best_fitness_val = float('inf')
    if final_fitness_values: 
        try:
            final_best_idx = min(range(len(final_fitness_values)), key=final_fitness_values.__getitem__)
            final_best_fitness_val = final_fitness_values[final_best_idx]
            if final_best_fitness_val < best_overall_fitness: # Comparar con el mejor histórico
                best_overall_fitness = final_best_fitness_val
                best_overall_solution = population[final_best_idx]
        except ValueError: 
             pass 
         
    return best_overall_solution, best_overall_fitness, best_fitness_history


# --- 4. Experimentación ---
# Definición de las 4 configuraciones
CONFIGURATIONS = [
    {'name': 'Base_Equilibrada', 'pop_size': 50, 'num_generations': 100, 'crossover_rate': 0.8, 'mutation_rate_per_gene': 0.1, 'mutation_strength_factor': 0.1, 'tournament_k': 3, 'elitism_count': 1},
    {'name': 'Mayor_Exploracion', 'pop_size': 100, 'num_generations': 150, 'crossover_rate': 0.7, 'mutation_rate_per_gene': 0.15, 'mutation_strength_factor': 0.15, 'tournament_k': 3, 'elitism_count': 2},
    {'name': 'Mayor_Explotacion', 'pop_size': 50, 'num_generations': 200, 'crossover_rate': 0.9, 'mutation_rate_per_gene': 0.05, 'mutation_strength_factor': 0.05, 'tournament_k': 5, 'elitism_count': 1},
    {'name': 'Rapida_Agresiva', 'pop_size': 30, 'num_generations': 80, 'crossover_rate': 0.85, 'mutation_rate_per_gene': 0.02, 'mutation_strength_factor': 0.2, 'tournament_k': 2, 'elitism_count': 1},
]

TARGET_FUNCTIONS = [f1, f2, f3, f4]
NUM_RUNS = 10 

output_dir = "ga_convergence_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results_summary_file = os.path.join(output_dir, "results_summary.txt")
with open(results_summary_file, "w") as f_summary:
    f_summary.write("Función, Configuración, Mejor Fitness Promedio, Mejor Fitness StdDev, Mejor Fitness Absoluto, Mejor Solución Absoluta\n")

    for func_obj in TARGET_FUNCTIONS:
        func_name = func_obj.__name__
        
        print(f"\n--- Optimizando {func_name} ---")
        f_summary.write(f"\n--- Optimizando {func_name} ---\n")
        specs = FUNCTION_SPECS[func_obj]
        
        for config in CONFIGURATIONS:
            config_name = config['name']
            print(f"\n  Configuración: {config_name}")
            f_summary.write(f"  Configuración: {config_name}\n")
            
            all_run_best_fitnesses = []
            all_run_best_solutions = []
            all_run_histories = []

            for run in range(NUM_RUNS):
                print(f"    Run {run+1}/{NUM_RUNS} para {func_name} con {config_name}")
                best_solution, best_fitness, history = genetic_algorithm(
                    obj_function=func_obj,
                    specs=specs,
                    pop_size=config['pop_size'],
                    num_generations=config['num_generations'],
                    crossover_rate=config['crossover_rate'],
                    mutation_rate_per_gene=config['mutation_rate_per_gene'],
                    mutation_strength_factor=config['mutation_strength_factor'],
                    tournament_k=config['tournament_k'],
                    elitism_count=config['elitism_count']
                )
                all_run_best_fitnesses.append(best_fitness)
                all_run_best_solutions.append(best_solution if best_solution is not None else [np.nan]*specs['dim'])
                all_run_histories.append(history)

                sol_str = f"[{', '.join(f'{s:.4f}' for s in best_solution)}]" if best_solution and len(best_solution) <=5 else f"{len(best_solution)}-dim vector" if best_solution else "N/A"
                print(f"      Run {run+1} Best Fitness: {best_fitness:.6f}, Solution: {sol_str}")

           
            valid_fitnesses = [f for f in all_run_best_fitnesses if f != float('inf') and not np.isnan(f)]
            
            if valid_fitnesses:
                avg_best_fitness = np.mean(valid_fitnesses)
                std_best_fitness = np.std(valid_fitnesses)
                overall_best_run_idx = np.argmin(all_run_best_fitnesses) 
                abs_best_fitness = all_run_best_fitnesses[overall_best_run_idx]
                abs_best_solution = all_run_best_solutions[overall_best_run_idx]
            else:
                avg_best_fitness = float('inf')
                std_best_fitness = float('nan')
                abs_best_fitness = float('inf')
                abs_best_solution = [np.nan] * specs['dim']

            
            print(f"\n  Resultados para {config_name} en {func_name}:")
            print(f"    Mejor Fitness promedio en {NUM_RUNS} runs: {avg_best_fitness:.6f} (std: {std_best_fitness:.6f})")
            print(f"    Mejor Fitness absoluto en {NUM_RUNS} runs: {abs_best_fitness:.6f}")
            sol_str_abs = f"[{', '.join(f'{s:.4f}' for s in abs_best_solution)}]" if abs_best_solution and not any(np.isnan(s) for s in abs_best_solution) and len(abs_best_solution) <=5 else f"{len(abs_best_solution)}-dim vector" if abs_best_solution else "N/A"
            print(f"    Mejor Solución absoluta: {sol_str_abs}")

            f_summary.write(f"    Mejor Fitness promedio: {avg_best_fitness:.6f}, StdDev: {std_best_fitness:.6f}\n")
            f_summary.write(f"    Mejor Fitness absoluto: {abs_best_fitness:.6f}\n")
            f_summary.write(f"    Mejor Solución absoluta: {sol_str_abs}\n\n")


            # Gráfico de convergencia
            plt.figure(figsize=(10, 6))
            for run_idx, history in enumerate(all_run_histories):
                if history and not all(h == float('inf') for h in history):
                    plt.plot(history, alpha=0.3, label=f"Run {run_idx+1}" if NUM_RUNS <= 10 else None) 

            # Opcional: Destacar la mejor run (la que alcanzó el mejor fitness final)
            if valid_fitnesses:
                 best_run_history = all_run_histories[overall_best_run_idx]
                 if best_run_history and not all(h == float('inf') for h in best_run_history):
                    plt.plot(best_run_history, color='red', linewidth=2, label=f"Mejor Run (Fitness: {abs_best_fitness:.4f})")
            
            

            plt.title(f"Convergencia GA - {func_name} - Config: {config_name}")
            plt.xlabel("Generación")
            plt.ylabel("Mejor Fitness Acumulado")
            if NUM_RUNS <= 10 or valid_fitnesses : plt.legend() 
            plt.grid(True)
            plt.yscale('symlog', linthresh=0.01) 
            
            plot_filename = os.path.join(output_dir, f"convergence_{func_name}_{config_name}.png")
            plt.savefig(plot_filename)
            print(f"    Gráfico guardado en: {plot_filename}")
            plt.close() 

print(f"\nProceso completado. Resultados resumidos en: {results_summary_file}")
print(f"Gráficos de convergencia guardados en la carpeta: {output_dir}")