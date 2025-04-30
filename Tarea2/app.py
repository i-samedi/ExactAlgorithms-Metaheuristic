import numpy as np
import os

def read_case(case):
    """
    - D : número de aviones (DOM).
    - E, P, L : arrays  de tiempo (temprano, prefetente, tardio).
    - Ci, Ck : arrays de costos por penalización por unidad bajo y sobre el prefetente.
    ..
    - tau : tiempo de separación minimos entre el aterrizaje minimo de dos aviones -> Tij.
    """
    with open(case, 'r') as f:
        D = int(f.readline())
        E, P, L, Ci, Ck = [], [], [], [], []
    for _ in range(D):
        e, p, l, ci, ck = f.readline().split()
        E. append(int(e)); P. append(int(p)); L. append(int(l)); Ci. append(float(ci)); Ck. append(float(ck))
    tau = [list(map(int, f.readline().split())) for _ in range(D)]
    return (D, np.array(E), np.array(P), np.array(L), np.array(Ci), np.array(Ck), np.array(tau))

if __name__ == '__main__':
    case_path = 'casos/case1.txt'
    print(f"Looking for file at: {case_path}")
    try:
        D, E, P, L, Ci, Ck, tau = read_case(case_path)
        print("retorno exitosamente")
        
        # Print some basic information to verify the data was read correctly
        print(f"Número de aviones: {D}")
        print(f"Tiempos tempranos: {E}")
        print(f"Tiempos preferentes: {P}")
        print(f"Tiempos tardíos: {L}")
        print(f"Costos por debajo: {Ci}")
        print(f"Costos por encima: {Ck}")
        print(f"Matriz de separación mínima (tau):")
        print(tau)
    except FileNotFoundError:
        print(f"¡Error! No se pudo encontrar el archivo en {case_path}")