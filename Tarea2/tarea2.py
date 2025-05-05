import numpy as np
import os
import sys

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

if __name__ == '__main__':
    #filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case1.txt'
    
    select = int(input("Selecciona el caso a evaluar: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n"))
    if select == 1: filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case1.txt'
    elif select == 2: filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case2.txt'
    elif select == 3: filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case3.txt' 
    elif select == 4: filename = '/Users/samedi/Documents/GitHub/Tareas-ExactAlgorithms-Metaheuristic/Tarea2/casos/case4.txt'
    else: sys.exit()
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    print(f"Buscando el archivo en: {file_path}")

    num_planes, earliest, preferred, latest, early_cost, late_cost, separation = read_file(file_path)
    print("-" * 20)
    print("Lectura del archivo exitosa.")
    print("-" * 20)

    print(f"Número de aviones (D): {num_planes}")
    print(f"Tiempos tempranos (E): {earliest}")
    print(f"Tiempos preferentes (P): {preferred}")
    print(f"Tiempos tardíos (L): {latest}")
    print(f"Costos penalización temprana (Ci/alpha): {early_cost}")
    print(f"Costos penalización tardía (Ck/beta): {late_cost}")
    print(f"Matriz de separación mínima (tau) [{separation.shape}]:")
    with np.printoptions(linewidth=np.inf):
        print(separation)
    print("-" * 20)