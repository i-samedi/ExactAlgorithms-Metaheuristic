import numpy as np
import os

def read_case(filepath):

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines() # Leer todas las líneas de una vez

        current_line_index = 0

        # Leer el número de aviones
        if not lines:
             raise ValueError("El archivo está vacío.")
        D = int(lines[current_line_index].strip())
        current_line_index += 1

        E, P, L, Ci, Ck = [], [], [], [], []
        tau_list = [] # Lista para ir guardando las filas de tau

        # --- Bucle principal corregido ---
        for k in range(D): # Iterar para cada avión k
            # 1. Leer los datos del avión k (E, P, L, Ci, Ck)
            if current_line_index >= len(lines):
                raise ValueError(f"Fin de archivo inesperado al buscar datos del avión {k+1}.")
            
            parts = lines[current_line_index].strip().split()
            if len(parts) != 5:
                # Error si la línea no tiene los 5 valores esperados para el avión
                raise ValueError(f"Error en formato de línea {current_line_index + 1} para avión {k+1}: Se esperaban 5 valores (E, P, L, Ci, Ck), se encontraron {len(parts)} -> {parts}")
            
            e, p, l, ci, ck = parts
            E.append(int(e))
            P.append(int(p))
            L.append(int(l))
            Ci.append(float(ci))
            Ck.append(float(ck))
            current_line_index += 1 # Avanzar a la siguiente línea (inicio de tau para avión k)

            # 2. Leer los tiempos de separación para el avión k (tau[k, :])
            tau_k_values = []
            while len(tau_k_values) < D:
                if current_line_index >= len(lines):
                     raise ValueError(f"Fin de archivo inesperado al leer la matriz tau para el avión {k+1}.")
                
                line_parts = lines[current_line_index].strip().split()
                # Asegurarse que los elementos leídos sean números antes de convertir
                try:
                   tau_k_values.extend(list(map(int, line_parts)))
                except ValueError:
                   raise ValueError(f"Error: Valor no numérico encontrado en la línea {current_line_index + 1} al leer tau para el avión {k+1}: {line_parts}")

                current_line_index += 1 # Avanzar a la siguiente línea para seguir leyendo tau si es necesario
            
            # Verificar si se leyeron exactamente D valores para la fila k de tau
            if len(tau_k_values) != D:
                 raise ValueError(f"Error: Se leyeron {len(tau_k_values)} valores para la fila {k+1} de tau, se esperaban {D}.")
            
            tau_list.append(tau_k_values) # Añadir la fila leída a la lista de tau

        # --- Fin del bucle principal ---

        # Convertir listas a arrays de NumPy al final
        E = np.array(E)
        P = np.array(P)
        L = np.array(L)
        Ci = np.array(Ci)
        Ck = np.array(Ck)
        tau = np.array(tau_list)

        # Verificación final de dimensiones (opcional pero útil)
        if E.shape != (D,) or P.shape != (D,) or L.shape != (D,) or \
           Ci.shape != (D,) or Ck.shape != (D,) or tau.shape != (D, D):
            raise ValueError("Error: Las dimensiones de los arrays leídos no coinciden con D.")

        return (D, E, P, L, Ci, Ck, tau)

    except FileNotFoundError:
        print(f"¡Error! No se pudo encontrar el archivo en {filepath}")
        raise
    except ValueError as ve:
        print(f"¡Error de formato o valor en el archivo {filepath}!: {ve}")
        raise
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer {filepath}: {e}")
        raise

# --- El bloque __main__ se mantiene igual que antes ---
if __name__ == '__main__':
    case_filename = 'casos/case1.txt'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_path = os.path.join(script_dir, case_filename)

    print(f"Buscando el archivo en: {case_path}")

    try:
        D, E, P, L, Ci, Ck, tau = read_case(case_path) # Usar la función corregida
        print("-" * 20)
        print("Lectura del archivo exitosa.")
        print("-" * 20)

        print(f"Número de aviones (D): {D}")
        print(f"Tiempos tempranos (E): {E}")
        print(f"Tiempos preferentes (P): {P}")
        print(f"Tiempos tardíos (L): {L}")
        print(f"Costos penalización temprana (Ci/alpha): {Ci}")
        print(f"Costos penalización tardía (Ck/beta): {Ck}")
        print(f"Matriz de separación mínima (tau) [{tau.shape}]]:")
        with np.printoptions(linewidth=np.inf):
             print(tau)
        print("-" * 20)

    except Exception as e:
        # El error específico ya se imprime dentro de read_case
        print(f"No se pudo procesar el caso.")