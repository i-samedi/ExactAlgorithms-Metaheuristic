import time

# Lista de comunas de la región de Brisketiana C[1,15]
comunas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Lista de Variables (representa la decisión de contruir o no un centro de vacunación)
variables = [f"C{comuna}" for comuna in comunas]
print("Variables:", variables)

dominios = {f"C{comuna}": [0, 1] for comuna in comunas}
print(f"Dominio de cada comuna de la región: ")
for comuna in dominios:
    print(f"{comuna}: {dominios[comuna]}")

costos = {
    "C1": 60, "C2": 30, "C3": 60, "C4": 70, "C5": 130,
    "C6": 60, "C7": 70, "C8": 60, "C9": 80, "C10": 70, 
    "C11": 50, "C12": 90, "C13": 30, "C14": 30, "C15": 100
}
print("Costos de construccion de centro de vacunación por cada comunas: ")
for comuna in costos:
    print(f"{comuna}: {costos[comuna]}")

cobertura = {
    "C1": ["C1", "C2", "C3", "C4", "C13"],
    "C2": ["C2", "C1", "C4", "C12", "C15"],
    "C3": ["C3", "C1", "C4", "C5", "C6", "C13"],
    "C4": ["C4", "C1", "C2", "C3", "C5", "C12"],
    "C5": ["C5", "C3", "C4", "C6", "C7", "C8", "C9", "C12"],
    "C6": ["C6", "C3", "C5", "C9"],           
    "C7": ["C7", "C5", "C8", "C10", "C11", "C12", "C14", "C15"],      
    "C8": ["C8", "C5", "C7", "C9", "C10"],
    "C9": ["C9", "C5", "C6", "C8", "C10", "C11"],
    "C10": ["C10", "C7", "C8", "C9", "C11"],
    "C11": ["C11", "C7", "C9", "C10", "C14"],
    "C12": ["C12", "C2", "C4", "C5", "C7", "C15"],
    "C13": ["C13", "C1", "C3"],
    "C14": ["C14", "C7", "C11", "C15"],
    "C15": ["C15", "C2", "C7", "C12", "C14"]        
}
print("Cantidad de comunas cubiertas por cada comuna: ")
for comuna in cobertura:
    cantidad = len(cobertura[comuna])
    print(f"{comuna}: {cantidad} comunas - {cobertura[comuna]}")
    
#------ Algoritmo Sin Heurística
def backtracking_sin_heuristica(asignacion, variables_no_asignadas, S_j, costo_actual, mejor_solucion, mejor_costo):
    if not S_j:  # Si todas las comunas están cubiertas
        if costo_actual < mejor_costo[0]:
            mejor_solucion[0] = asignacion.copy()
            mejor_costo[0] = costo_actual
        return

    if not variables_no_asignadas:  # Si no quedan variables por asignar
        return

    Ci = variables_no_asignadas[0]  # Tomamos la primera variable no asignada
    nuevas_variables = variables_no_asignadas[1:]

    # Asignar Ci = 1 (construir)
    asignacion[Ci] = 1
    nuevo_S_j = {Cj: S_j[Cj].copy() for Cj in S_j if Cj not in cobertura[Ci]}
    backtracking_sin_heuristica(asignacion, nuevas_variables, nuevo_S_j, costo_actual + costos[Ci], mejor_solucion, mejor_costo)

    # Asignar Ci = 0 (no construir)
    asignacion[Ci] = 0
    nuevo_S_j = {Cj: S_j[Cj].copy() for Cj in S_j}
    consistente = True
    for Cj in cobertura[Ci]:
        if Cj in nuevo_S_j:
            nuevo_S_j[Cj].remove(Ci)
            if not nuevo_S_j[Cj]:
                consistente = False
                break
    if consistente:
        backtracking_sin_heuristica(asignacion, nuevas_variables, nuevo_S_j, costo_actual, mejor_solucion, mejor_costo)
    
    del asignacion[Ci]  # Limpiar asignación para backtracking
    
asignacion = {}
variables_no_asignadas = variables.copy()  # Lista para mantener el orden
S_j = {f"C{j}": {Ci for Ci in variables if f"C{j}" in cobertura[Ci]} for j in comunas}
mejor_solucion = [None]
mejor_costo = [float('inf')]

tiempo_inicio = time.time()
backtracking_sin_heuristica(asignacion, variables_no_asignadas, S_j, 0, mejor_solucion, mejor_costo)

tiempo_fin = time.time()
tiempo_ejecucion = tiempo_fin - tiempo_inicio

print("Mejor solución encontrada (sin heurística):")
for Ci, valor in mejor_solucion[0].items():
    if valor == 1:
        print(f"Construir en {Ci} (costo: {costos[Ci]})")
print(f"\n-> Costo total: '{mejor_costo[0]}'")
print(f"-> Tiempo de ejecución: '{tiempo_ejecucion:4f}' segundos")

#------ Algoritmo Con Heurística
def greedy_cobertura(variables, costos, cobertura):
    # Conjunto de comunas no cubiertas
    no_cubiertas = set(variables)
    # Solución: comunas donde se construirán centros
    solucion = {}
    costo_total = 0

    while no_cubiertas:
        mejor_comuna = None
        mejor_eficiencia = -1

        # Evaluamos cada comuna candidata
        for comuna in variables:
            if comuna not in solucion:  # Solo comunas no seleccionadas
                cubre_nuevas = [c for c in cobertura[comuna] if c in no_cubiertas]
                num_cubre_nuevas = len(cubre_nuevas)
                if num_cubre_nuevas > 0:
                    eficiencia = num_cubre_nuevas / costos[comuna]
                    if eficiencia > mejor_eficiencia:
                        mejor_eficiencia = eficiencia
                        mejor_comuna = comuna
                        mejores_cubre_nuevas = cubre_nuevas

        # Seleccionamos la mejor comuna
        solucion[mejor_comuna] = 1
        costo_total += costos[mejor_comuna]
        # Actualizamos las comunas no cubiertas
        for c in mejores_cubre_nuevas:
            no_cubiertas.remove(c)

    return solucion, costo_total

# Ejecutamos el algoritmo
tiempo_inicio = time.time()

solucion, costo = greedy_cobertura(variables, costos, cobertura)

tiempo_fin = time.time()
tiempo_ejecucion_2 = tiempo_fin - tiempo_inicio

# Mostramos los resultados
print("Solución Greedy:")
for comuna in solucion:
    print(f"Construir en {comuna} (costo: {costos[comuna]})")
print(f"\n-> Costo total: '{costo}'")
print(f"-> Tiempo de ejecución: '{tiempo_ejecucion_2:4f}'segundos")


# Configuración para Jupyter Notebook
#%pip install matplotlib numpy
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Configurar el primer eje (costo)
algoritmos = ['Backtracking\n(sin heurística)', 'Greedy\n(con heurística)']
x = np.arange(len(algoritmos))
width = 0.35

# Barras para el costo (eje y principal)
rects1 = ax1.bar(x - width/2, [mejor_costo[0], costo], width, label='Costo', color='skyblue')  # Corregir mejor_costo
ax1.set_ylabel('Costo total', fontsize=12, color='navy')
ax1.set_ylim(0, max(mejor_costo[0], costo) * 1.2)
ax1.tick_params(axis='y', labelcolor='navy')

# Crear un segundo eje y para el tiempo de ejecución
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, [tiempo_ejecucion, tiempo_ejecucion_2], width, label='Tiempo', color='coral')
ax2.set_ylabel('Tiempo de ejecución (segundos)', fontsize=12, color='darkred')
ax2.set_ylim(0, max(tiempo_ejecucion, tiempo_ejecucion_2) * 1.2)
ax2.tick_params(axis='y', labelcolor='darkred')

# Añadir etiquetas en las barras
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# Configurar eje x
ax1.set_xticks(x)
ax1.set_xticklabels(algoritmos, fontsize=11)

# Títulos y leyenda
plt.title('Comparación de Costo vs Tiempo de Ejecución', fontsize=14, pad=20)
fig.tight_layout()

# Agregar leyendas
linea1 = plt.Line2D([0], [0], color='skyblue', lw=4)
linea2 = plt.Line2D([0], [0], color='coral', lw=4)
plt.legend([linea1, linea2], ['Costo total', 'Tiempo (segundos)'], loc='upper left')

# Agregar grid para mejor legibilidad
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Añadir contexto con texto
plt.figtext(0.5, 0.01, 'Comparación de rendimiento para la cobertura de comunas en Brisketiana', 
            ha='center', fontsize=10, fontstyle='italic')

plt.show()
