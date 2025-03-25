# Datos iniciales
comunas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
variables = [f"C{comuna}" for comuna in comunas]
dominios = {f"C{comuna}": [0, 1] for comuna in comunas}
costos = {
    "C1": 60, "C2": 30, "C3": 60, "C4": 70, "C5": 130,
    "C6": 60, "C7": 70, "C8": 60, "C9": 80, "C10": 70, 
    "C11": 50, "C12": 90, "C13": 30, "C14": 30, "C15": 100
}
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

# Algoritmo de backtracking con Forward Checking (sin heurística)
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

# Inicialización
asignacion = {}
variables_no_asignadas = variables.copy()  # Lista para mantener el orden
S_j = {f"C{j}": {Ci for Ci in variables if f"C{j}" in cobertura[Ci]} for j in comunas}
mejor_solucion = [None]
mejor_costo = [float('inf')]

# Ejecutar el algoritmo
backtracking_sin_heuristica(asignacion, variables_no_asignadas, S_j, 0, mejor_solucion, mejor_costo)

# Resultados
print("Mejor solución encontrada (sin heurística):")
for Ci, valor in mejor_solucion[0].items():
    if valor == 1:
        print(f"Construir en {Ci} (costo: {costos[Ci]})")
print(f"Costo total: {mejor_costo[0]}")