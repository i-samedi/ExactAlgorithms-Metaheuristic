
import numpy as np
import os
import sys
import random # Needed for stochastic greedy

def read_file(path):
    """
    Reads the aircraft landing problem data from a file.
    - D : número de aviones (DOM).
    - E, P, L : arrays de tiempo (temprano, preferente, tardio).
    - Ci, Ck : arrays de costos por penalización por unidad bajo y sobre el preferente.
    - tau : tiempo de separación minimos entre el aterrizaje minimo de dos aviones -> Tij.
             tau[i][j] is the separation needed AFTER plane i lands BEFORE plane j can land.
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
            Ci.append(float(values[3])) # alpha in some literature (cost if early)
            Ck.append(float(values[4])) # beta in some literature (cost if late)
            index += 1

            # Read separation times for plane i
            sep_row = []
            while len(sep_row) < num_planes:
                 # Check if index is out of bounds
                if index >= len(data):
                     raise ValueError(f"Error parsing separation times for plane {i+1}: Unexpected end of file.")
                line_values = data[index].strip().split()
                 # Handle potential empty lines
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
        
        # Basic validation
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
    Ci = np.array(Ci) # Cost per unit time *EARLY*
    Ck = np.array(Ck) # Cost per unit time *LATE*
    tau = np.array(tau_list)

    # Validate E <= P <= L
    if not np.all(E <= P) or not np.all(P <= L):
        print("Warning: Not all planes satisfy E <= P <= L.")
        # Decide if this should be a fatal error or just a warning
        # For now, just print warning and continue

    return num_planes, E, P, L, Ci, Ck, tau

def calculate_cost(schedule, P, Ci, Ck):
    """Calculates the total cost of a given landing schedule."""
    total_cost = 0
    scheduled_planes = set()
    
    if not schedule:
        return 0

    for plane_idx, landing_time in schedule:
        if plane_idx in scheduled_planes:
             print(f"Warning: Plane {plane_idx} scheduled multiple times in cost calculation.")
             continue # Avoid double counting cost if error in schedule generation
        scheduled_planes.add(plane_idx)

        preferred_time = P[plane_idx]
        cost_early = Ci[plane_idx]
        cost_late = Ck[plane_idx]

        if landing_time < preferred_time:
            total_cost += cost_early * (preferred_time - landing_time)
        elif landing_time > preferred_time:
            total_cost += cost_late * (landing_time - preferred_time)
        # If landing_time == preferred_time, cost is 0

    return total_cost

def greedy_deterministic(num_planes, E, P, L, Ci, Ck, tau):
    """
    Implements a deterministic greedy algorithm for ALP.
    Strategy: Sort planes by Earliest landing time (E_k) and schedule them
              sequentially at the earliest possible valid time.
    Assumes a single runway.
    """
    # Sort plane indices based on their earliest landing time E_k
    plane_indices = list(range(num_planes))
    # Sort by E, then P as a tie-breaker (optional but can help consistency)
    sorted_indices = sorted(plane_indices, key=lambda k: (E[k], P[k]))

    schedule = []
    landed_times = {} # Store landing times {plane_idx: time}
    last_plane_idx = -1
    last_landing_time = -1

    for k in sorted_indices:
        # Determine earliest possible landing time for plane k
        min_allowed_time = E[k]

        # Check separation constraint with the immediately preceding plane in the sequence
        if last_plane_idx != -1:
            separation_needed = tau[last_plane_idx][k]
            # Handle potential large dummy values if needed (like 99999)
            if separation_needed > L[k]: # Arbitrarily large value likely means cannot follow
                 # This might indicate an issue if 99999 isn't just for i==j
                 pass # Assume large values only for i==j based on example
            
            # Earliest time based on separation from LAST plane landed
            earliest_after_separation = last_landing_time + separation_needed
            min_allowed_time = max(min_allowed_time, earliest_after_separation)

        # Actual landing time is the earliest possible time
        actual_landing_time = min_allowed_time

        # Check feasibility constraint (Latest landing time L_k)
        if actual_landing_time > L[k]:
            print(f"Warning: Deterministic greedy resulted in infeasible time for plane {k}. Required: {actual_landing_time}, Latest: {L[k]}. Sequence: {[p[0] for p in schedule] + [k]}")
            # Handle infeasibility: return high cost, partial schedule, or raise error
            # For now, let's return the partial schedule and its cost, maybe add a flag
            return schedule, calculate_cost(schedule, P, Ci, Ck), False # False indicates infeasibility

        # Add to schedule
        schedule.append((k, actual_landing_time))
        landed_times[k] = actual_landing_time
        last_plane_idx = k
        last_landing_time = actual_landing_time

    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, total_cost, True # True indicates feasible schedule found

def greedy_stochastic(num_planes, E, P, L, Ci, Ck, tau, rcl_size=3, seed=None):
    """
    Implements a stochastic greedy algorithm for ALP using RCL.
    Strategy: At each step, build a Restricted Candidate List (RCL) of planes
              that can land next feasibly, based on the earliest possible landing time.
              Randomly select a plane from the RCL to schedule next.
    Assumes a single runway.
    """
    if seed is not None:
        random.seed(seed)

    unscheduled_planes = set(range(num_planes))
    schedule = []
    landed_times = {}
    last_plane_idx = -1
    last_landing_time = -1

    while unscheduled_planes:
        candidates = [] # List of (plane_idx, earliest_possible_time)

        for k in unscheduled_planes:
            # Determine earliest possible landing time for plane k if scheduled NOW
            min_allowed_time = E[k]

            if last_plane_idx != -1:
                separation_needed = tau[last_plane_idx][k]
                earliest_after_separation = last_landing_time + separation_needed
                min_allowed_time = max(min_allowed_time, earliest_after_separation)

            # Check if this plane *can* be scheduled next (respecting L_k)
            if min_allowed_time <= L[k]:
                candidates.append((k, min_allowed_time))

        if not candidates:
            # No plane can be scheduled next, possible infeasibility or error
            print(f"Warning: Stochastic greedy could not schedule any remaining plane. Unscheduled: {unscheduled_planes}. Last landed: {last_plane_idx} at {last_landing_time}")
            # Return partial schedule
            return schedule, calculate_cost(schedule, P, Ci, Ck), False

        # Build RCL: Sort candidates by earliest possible time and take top 'rcl_size'
        candidates.sort(key=lambda x: x[1]) # Sort by earliest_possible_time
        
        # Determine actual size of RCL (can be smaller than rcl_size if fewer candidates)
        current_rcl_size = min(rcl_size, len(candidates))
        rcl = candidates[:current_rcl_size]

        # Randomly select one plane from the RCL
        chosen_plane_idx, chosen_landing_time = random.choice(rcl)

        # Add to schedule
        schedule.append((chosen_plane_idx, chosen_landing_time))
        landed_times[chosen_plane_idx] = chosen_landing_time
        last_plane_idx = chosen_plane_idx
        last_landing_time = chosen_landing_time
        unscheduled_planes.remove(chosen_plane_idx)

    total_cost = calculate_cost(schedule, P, Ci, Ck)
    return schedule, total_cost, True


if __name__ == '__main__':
    
    DEFAULT_CASE_DIR = "casos/" # Adjust if your cases are elsewhere

    if len(sys.argv) > 1:
        # Use command line argument if provided
        select = sys.argv[1]
        if select.isdigit():
             case_num = int(select)
             if 1 <= case_num <= 4:
                 filename = os.path.join(DEFAULT_CASE_DIR, f"case{case_num}.txt")
             else:
                 print("Invalid case number from argument. Choose 1-4.")
                 sys.exit(1)
        else:
             # Assume argument is a full path
             filename = select 
    else:
        # Interactive selection
        select = input("Selecciona el caso a evaluar: \n-> 1. Caso 1\n-> 2. Caso 2\n-> 3. Caso 3\n-> 4. Caso 4\n-> 0. Salir\n")
        if select == '1': filename = os.path.join(DEFAULT_CASE_DIR, 'case1.txt')
        elif select == '2': filename = os.path.join(DEFAULT_CASE_DIR, 'case2.txt')
        elif select == '3': filename = os.path.join(DEFAULT_CASE_DIR, 'case3.txt')
        elif select == '4': filename = os.path.join(DEFAULT_CASE_DIR, 'case4.txt')
        elif select == '0': sys.exit()
        else: 
             print("Selección inválida.")
             sys.exit()

    # Use absolute path for clarity if needed, especially for nested directories
    if not os.path.isabs(filename):
         script_dir = os.path.dirname(os.path.abspath(__file__))
         file_path = os.path.join(script_dir, filename)
    else:
         file_path = filename

    print(f"--- Reading data from: {file_path} ---")
    num_planes, E, P, L, Ci, Ck, tau = read_file(file_path)
    print("-" * 20)
    print("Lectura del archivo exitosa.")
    print(f"Número de aviones (D): {num_planes}")
    # Optional: Print other details if needed for debugging
    # print(f"Tiempos tempranos (E): {E}")
    # print(f"Tiempos preferentes (P): {P}")
    # print(f"Tiempos tardíos (L): {L}")
    # print(f"Costos penalización temprana (Ci): {Ci}")
    # print(f"Costos penalización tardía (Ck): {Ck}")
    # print(f"Matriz de separación mínima (tau) [{tau.shape}]:")
    # with np.printoptions(linewidth=np.inf):
    #     print(tau)
    print("-" * 20)

    # --- Run Deterministic Greedy ---
    print("--- Running Deterministic Greedy (Sort by Earliest Time E_k) ---")
    det_schedule, det_cost, det_feasible = greedy_deterministic(num_planes, E, P, L, Ci, Ck, tau)
    print(f"Feasible: {det_feasible}")
    print(f"Total Cost: {det_cost:.2f}")
    # Format schedule for readability: sort by time, show index+1
    det_schedule_formatted = sorted([(p[0] + 1, p[1]) for p in det_schedule], key=lambda x: x[1])
    print(f"Schedule (Plane #, Time): {det_schedule_formatted}")
    print("-" * 20)


    # --- Run Stochastic Greedy (10 times) ---
    print("--- Running Stochastic Greedy (RCL based on Earliest Possible Time) ---")
    num_stochastic_runs = 10
    rcl_size_param = 3 # Example parameter, can be tuned
    print(f"Using RCL size = {rcl_size_param}")
    
    stochastic_results = []
    for i in range(num_stochastic_runs):
        seed = i # Use loop index as seed for reproducibility
        sto_schedule, sto_cost, sto_feasible = greedy_stochastic(num_planes, E, P, L, Ci, Ck, tau, rcl_size=rcl_size_param, seed=seed)
        stochastic_results.append({'seed': seed, 'cost': sto_cost, 'feasible': sto_feasible, 'schedule': sto_schedule})
        
        # Format schedule for printing: sort by time, show index+1
        sto_schedule_formatted = sorted([(p[0] + 1, p[1]) for p in sto_schedule], key=lambda x: x[1])
        
        print(f"\nRun {i+1} (Seed: {seed})")
        print(f"Feasible: {sto_feasible}")
        print(f"Total Cost: {sto_cost:.2f}")
        # print(f"Schedule (Plane #, Time): {sto_schedule_formatted}") # Keep output concise

    print("-" * 20)
    print("--- Stochastic Greedy Summary ---")
    costs = [res['cost'] for res in stochastic_results if res['feasible']]
    if costs:
        print(f"Best Cost: {min(costs):.2f}")
        print(f"Average Cost: {np.mean(costs):.2f}")
        print(f"Worst Cost: {max(costs):.2f}")
        print(f"Standard Deviation: {np.std(costs):.2f}")
        num_infeasible = sum(1 for res in stochastic_results if not res['feasible'])
        if num_infeasible > 0:
             print(f"Number of infeasible runs: {num_infeasible}/{num_stochastic_runs}")
    else:
        print("All stochastic runs resulted in infeasible solutions.")
    print("-" * 20)

