### OR HACKATHON, Sponsored By Califrais and Cermics research lab ###

import pandas as pd 
from math import *
import numpy as np 
import time 

# Importing and reading vehicle family params and instances files as dataframes

vehicles_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/vehicles.csv'
instance1_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_01.csv'
instance2_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_02.csv'
instance3_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_03.csv'
instance4_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_04.csv'
instance5_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_05.csv'
instance6_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_06.csv'
instance7_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_07.csv'
instance8_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_08.csv'
instance9_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_09.csv'
instance10_path = '/Users/antoinechosson/Desktop/KIRO2025/instances/instance_10.csv'


vehicles = pd.read_csv(vehicles_path)
instance1 = pd.read_csv(instance1_path)
instance2 = pd.read_csv(instance2_path)
instance3 = pd.read_csv(instance3_path)
instance4 = pd.read_csv(instance4_path)
instance5 = pd.read_csv(instance5_path)
instance6 = pd.read_csv(instance6_path)
instance7 = pd.read_csv(instance7_path)
instance8 = pd.read_csv(instance8_path)
instance9 = pd.read_csv(instance9_path)
instance10 = pd.read_csv(instance10_path)

############################################################

# Time and space function 

def gamma(f,t):
    """
    Coef for influence of time on vehicle displacement speed 
    """
    res = 0
    w = (2*pi)/86400
    for n in range (0,4): 
        alpha_f_n = vehicles.iloc[f]['fourier_cos_'+ str(n)]
        beta_f_n = vehicles.iloc[f]['fourier_sin_'+ str(n)]
        res += alpha_f_n*cos(n*w*t) + beta_f_n*sin(n*w*t)
    return res

def convert_x(phi_i, phi_j): 
    """
    Convert latitude into x cartesian coordinate
    """
    ro = 6.371e6
    return ro*((2*pi)/360)*(phi_j-phi_i)


def convert_y(lambda_i, lambda_j): 
    """
    Convert latitude into y cartesian coordinate
    """
    ro = 6.371e6
    phi_0 = 48.764246
    return ro*(cos(((2*pi)/360)*phi_0))*((2*pi/360)*(lambda_j-lambda_i))


def travel_time(f,i,j,t, instance): 
    """
    Time for f type of vehicle to travel from i to j at timestamp t 
    """
    vehicle_idx = f - 1  

    phi_i, lambda_i = instance.iloc[i]['latitude'], instance.iloc[i]['longitude']
    phi_j, lambda_j = instance.iloc[j]['latitude'], instance.iloc[j]['longitude']

    speed_factor = gamma(vehicle_idx, t) 
    base_speed = vehicles.iloc[vehicle_idx]['speed']
    actual_speed = base_speed * speed_factor 

    manhattan_dist = abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 
    p_f = vehicles.iloc[vehicle_idx]['parking_time']

    return manhattan_dist/actual_speed + p_f

travel_cache = {}
def travel_time_fast(f, i, j, t, instance):
    """
    Optimization of travel time computation : 
    Reduces computation time by keeping in memory in a set the already
    calculated travel times 
    """
    key = (f, i, j, round(t, 2))  
    if key in travel_cache:
        return travel_cache[key]
    
    val = travel_time(f, i, j, t, instance)
    travel_cache[key] = val
    return val

def delta_m(i,j, instance): 
    """
    Manhattan distance between points i and j 
    """
    phi_i, lambda_i = instance.iloc[i]['latitude'], instance.iloc[i]['longitude']
    phi_j, lambda_j = instance.iloc[j]['latitude'], instance.iloc[j]['longitude']
    return abs(convert_x(phi_i, phi_j)) + abs(convert_y(lambda_i, lambda_j)) 

def delta_e(i,j, instance): 
    """
    Euclidian distance between points i and j 
    """
    phi_i, lambda_i = instance.iloc[i]['latitude'], instance.iloc[i]['longitude']
    phi_j, lambda_j = instance.iloc[j]['latitude'], instance.iloc[j]['longitude']
    return sqrt(abs(convert_x(phi_i, phi_j))**2 + abs(convert_y(lambda_i, lambda_j))**2)

def delta_M(instance): 
    """
    Computing matrix of Manhattan distances 
    Optimized using numpy vectorization
    """
    n = len(instance)
    lats = instance['latitude'].values
    lons = instance['longitude'].values
    
    # Creating a coordinate grids
    lat_i, lat_j = np.meshgrid(lats, lats, indexing='ij')
    lon_i, lon_j = np.meshgrid(lons, lons, indexing='ij')
    
    ro = 6.371e6
    phi_0 = 48.764246
    
    x_dist = ro * (2*np.pi/360) * (lat_j - lat_i)
    y_dist = ro * np.cos((2*np.pi/360)*phi_0) * (2*np.pi/360) * (lon_j - lon_i)
    
    return np.abs(x_dist) + np.abs(y_dist)

def delta_E(instance):
    """
    Computing matrix of Euclidian distances 
    Optimized using numpy vectorization
    """
    n = len(instance)
    lats = instance['latitude'].values
    lons = instance['longitude'].values
    
    lat_i, lat_j = np.meshgrid(lats, lats, indexing='ij')
    lon_i, lon_j = np.meshgrid(lons, lons, indexing='ij')
    
    ro = 6.371e6
    phi_0 = 48.764246
    
    x_dist = ro * (2*np.pi/360) * (lat_j - lat_i)
    y_dist = ro * np.cos((2*np.pi/360)*phi_0) * (2*np.pi/360) * (lon_j - lon_i)
    
    return np.sqrt(x_dist**2 + y_dist**2)



### GENERATING FEASIBLE SOLUTIONS ####
# R set of routes r realised by vehicle family f 

def is_feasible(route, f, instance):
    """
    Check if a route r is feasible for a given vehicle family f 
    """
    vehicle_idx = f-1
    n = len(route)

    # Constraint 1 : Vehicle needs to start and end at depot
    if route[0] != 0 or route[-1] != 0 : 
        return False 
    
    # Constraint 2 : Total order weight must fit in vehicle capacity 
    total_weight = 0
    for i in route[1:-1]:
        total_weight += instance.iloc[i]['order_weight']
    if total_weight > vehicles.iloc[vehicle_idx]['max_capacity']:
        return False 
    
    #### Time constraints ####

    d = 0 

    for k in range(n-1): 
        current_order = route[k]
        next_order = route[k+1]
        arrival_next = d + travel_time_fast(f, current_order, next_order, d, instance)
        
        # if we are at depot : route is finished, nothing to check 
        if next_order == 0 : 
            pass 
        else : 

            # getting the delivery times window and duration of delivery 
            start = instance.iloc[next_order]['window_start']
            end = instance.iloc[next_order]['window_end']
            delivery_duration = instance.iloc[next_order]['delivery_duration']

            # Constraint 3 : if arriving too early -> wait until start time
            if arrival_next < start :
                arrival_next = start 
            # Constraint 4 : Can't arriving too late 
            if arrival_next > end : 
                return False 
            
            # Constraint 5 : update duration d by adding delivery duration 
            d = arrival_next + delivery_duration
    
    return True 

def compute_arrival_times(route, f, instance):
    """
    Compute consecutive arrival times for a route executed by a vehicle of family f 
    """
    arrival = [0]
    t = 0

    for k in range(len(route)-1):
        i = route[k]
        j = route[k+1]

        arr = t + travel_time_fast(f, i, j, t, instance)

        if j != 0:
            start = instance.iloc[j]['window_start']
            end   = instance.iloc[j]['window_end']
            serv  = instance.iloc[j]['delivery_duration']

            if arr < start:
                arr = start
            if arr > end:
                return None  

            t = arr + serv
        else:
            t = arr  

        arrival.append(t)

    return arrival

def feasible_from(route, f, instance, start_idx, old_arr):
    """
    Optimized feasibility computation : 
    Computed feasibility of route only from start_idx onward
    Also returns the arrival times 
    """
    new_arr = old_arr[:start_idx+1]
    t = new_arr[-1]

    for k in range(start_idx, len(route)-1):
        i = route[k]
        j = route[k+1]

        arr = t + travel_time_fast(f, i, j, t, instance)

        if j != 0:
            start = instance.iloc[j]['window_start']
            end   = instance.iloc[j]['window_end']
            serv  = instance.iloc[j]['delivery_duration']

            if arr < start:
                arr = start
            if arr > end:
                return False, None

            t = arr + serv
        else:
            t = arr

        new_arr.append(t)

    return True, new_arr

def is_feasible_incremental(route, f, instance, arrival_times=None):
    """
    Feasibility check computation optimized using arrival times
    """
    # Capacity constraint 
    vehicle_idx = f-1
    total_weight = 0
    for i in route[1:-1]:
        total_weight += instance.iloc[i]['order_weight']
    if total_weight > vehicles.iloc[vehicle_idx]['max_capacity']:
        return False

    # Depot constraints
    if route[0] != 0 or route[-1] != 0:
        return False

    # Time constraints
    if arrival_times is None:
        arrival_times = compute_arrival_times(route, f, instance)
    
    return arrival_times is not None

def route_cost(route, f, instance, M, E): 
    """
    Computes objective function for a given route and car family 
    Total cost = rental cost + fuel cost + radius penalty cost 
    """
    vehicle_idx = f-1

    # Rental cost 
    c_rental = vehicles.iloc[vehicle_idx]['rental_cost']

    # Fuel cost 
    fuel_cost_per_meter = vehicles.iloc[vehicle_idx]['fuel_cost']
    c_fuel = 0 
    for k in range (len(route)-1): 
        c_fuel += fuel_cost_per_meter*M[route[k], route[k+1]]
    
    # Eucledian radius penalty 
    radius_cost = vehicles.iloc[vehicle_idx]['radius_cost']
    max_euclidian_distance = 0
    delivery_points = [i for i in route if i != 0]
    for i in range(len(delivery_points)):
        for j in range(i + 1, len(delivery_points)):
            a = delivery_points[i]
            b = delivery_points[j]
            max_euclidian_distance = max(max_euclidian_distance, E[a, b])
    c_radius = radius_cost*(0.5*max_euclidian_distance)

    return c_rental + c_fuel + c_radius 

def get_deliveries(instance):
    """
    Returns the delivery points of an instance outside of depot 
    """
    return list(range(1, len(instance)))


### Aggregating individual routes to create solution R ###

def solution_cost(R, instance, M, E):  
    """
    Computes total cost of a set of routes R
    Chosen format for solution R : 

    R = {
        0: {"family": 1, "route": [0, 12, 5, 19, 0]},
        1: {"family": 2, "route": [0, 8, 21, 7, 0]},
    }
    """
    tot_cost = 0
    for r in R: 
        f = R[r]['family']
        route = R[r]['route']
        tot_cost += route_cost(route, f, instance, M, E)
    return tot_cost 

def is_solution_feasible(R, instance): 
    """
    Checks if global solution R is feasible 
    """
    visited = set()
    for r in R : 
        f = R[r]['family']
        route = R[r]['route']
        if is_feasible(route, f, instance) == False : 
            return False, f"route {r} is infeasible"
    
        for delivery_point in route[1:-1]:
            if delivery_point in visited:
                return False, f"delivery {delivery_point} visited more than once"
            visited.add(delivery_point)

    all_deliveries = set(get_deliveries(instance))
    missing = all_deliveries - visited
    if len(missing) > 0:
        return False, f"Missing orders: {missing}"
    return True, "Solution is feasible"


#### Computing a first simple greedy heuristic ###

def next_feasible_node(previous_node, unvisited, f, current_route, instance, M):
    """
    Looks for nearest delivery point (feasible) to add to current route 
    """
    distances = []
    for node in unvisited:
        dist = M[previous_node][node] 
        distances.append((dist, node))
    distances.sort()
    
    # try each node from closest to farest 
    for dist, next_node in distances: 
        new_route = current_route[:-1] + [next_node, 0]
        if is_feasible(new_route, f, instance): 
            return next_node 
        
    return None 

def build_solution_with_family(f, instance, M, E):
    """
    Full solution building with integrated arrival times storage
    """
    R = {}
    r = 0
    unvisited = set(get_deliveries(instance))

    while unvisited:
        R[r] = {'family': f, 'route': [0, 0]}
        current_route = R[r]['route']

        while True:
            prev_delivery = current_route[-2]
            next_delivery = next_feasible_node(prev_delivery, unvisited, f, current_route, instance, M)
            
            if next_delivery is None:
                break

            current_route.insert(-1, next_delivery)
            unvisited.remove(next_delivery)
        
        # Store arrival times for this route
        arrival_times = compute_arrival_times(R[r]['route'], f, instance)
        if arrival_times is not None:
            R[r]['arrival'] = arrival_times
        else:
            # If route is infeasible, don't store arrival times
            print(f"Route {r} has no valid arrival times")
            
        r += 1
        
        # Prevent infinite loop
        if r > 1000:
            print("Too many routes created")
            break

    return R if R else None

def build_solution(instance, M, E):
    """
    Try different vehicle families to find the best solution
    """
    best_solution = None
    best_cost = float('inf')
    
    num_families = len(vehicles)
    
    for f in range(1, num_families + 1):
        try:
            R = build_solution_with_family(f, instance, M, E)
            if R is None or len(R) == 0:
                continue
                
            cost = solution_cost(R, instance, M, E)
            if cost < best_cost:
                best_cost = cost
                best_solution = R
        except Exception as e:
            continue
    
    # If no best solution found : try with 1 as default
    if best_solution is None:
        try:
            best_solution = build_solution_with_family(1, instance, M, E)
        except:
            print("Fallback failed")
    
    return best_solution


### Upgrading solution with relocation of nodes ###

def remove_node_from_route(route, node):
    """
    Removes node from a given route 
    """
    new_route = route.copy()
    new_route.remove(node)
    return new_route

def limited_insert_positions(route, node, M):
    """
    Insert node at positions near its nearest neighbor in the route
    """
    if len(route) <= 2:
        return [[0, node, 0]]

    deliveries = route[1:-1]
    if not deliveries: 
        return [[0, node, 0]]
    
    # Find nearest neighbor in the route
    nearest = min(deliveries, key=lambda x: M[node][x])
    idx = route.index(nearest)

    candidates = []
    # Try to insert before nearest, at nearest, after nearest
    for pos in [idx-1, idx, idx+1]:
        if 1 <= pos < len(route): 
            new_route = route[:pos] + [node] + route[pos:]
            candidates.append(new_route)
    
    return candidates

def compute_local_fuel_delta(route, node, pos, M, f):
    """
    Computes the variation in fuel consumption after node realocation 
    From node to pos in route
    """
    if pos == 0 or pos >= len(route):
        return float('inf')  # Invalid position
    
    prev = route[pos-1]
    next_node = route[pos]
    
    # Fuel cost per meter for this vehicle family
    fuel_cost_per_meter = vehicles.iloc[f-1]['fuel_cost']
    
    delta_fuel = (M[prev][node] + M[node][next_node] - M[prev][next_node]) * fuel_cost_per_meter
    
    return delta_fuel

def relocate_once(R, instance, M, E):
    """
    Optimized relocation with incremental feasibility checking
    """
    # For each pair of routes
    for r1 in R:
        route1 = R[r1]["route"]
        f1 = R[r1]["family"]
        delivery1 = route1[1:-1]
        
        if len(delivery1) == 0:
            continue

        for node in delivery1:
            new_r1 = remove_node_from_route(route1, node)
            if len(new_r1) < 2:
                continue

            # Check if removal is feasible with incremental check
            removal_pos = route1.index(node)
            if removal_pos > 0 and 'arrival' in R[r1]:
                feas_r1, new_arr_r1 = feasible_from(new_r1, f1, instance, removal_pos-1, R[r1]['arrival'])
                if not feas_r1:
                    continue
            else:
                # Full recompute if removing from start or no cached arrival times
                new_arr_r1 = compute_arrival_times(new_r1, f1, instance)
                if new_arr_r1 is None:
                    continue

            for r2 in R:
                if r1 == r2:
                    continue

                route2 = R[r2]["route"]
                f2 = R[r2]["family"]

                # Use limited insertion positions
                for cand in limited_insert_positions(route2, node, M):
                    
                    # Find insertion position
                    pos = -1
                    for i in range(1, len(route2)):
                        test_route = route2[:i] + [node] + route2[i:]
                        if test_route == cand:
                            pos = i
                            break
                    
                    if pos == -1:
                        continue
                    
                    # Quick fuel delta check
                    fuel_delta = compute_local_fuel_delta(route2, node, pos, M, f2)
                    if fuel_delta > 50:
                        continue
                    
                    # Incremental feasibility check for insertion
                    if 'arrival' in R[r2]:
                        feas_r2, new_arr_r2 = feasible_from(cand, f2, instance, pos-1, R[r2]['arrival'])
                        if not feas_r2:
                            continue
                    else:
                        # Full feasibility check if no cached arrival times
                        new_arr_r2 = compute_arrival_times(cand, f2, instance)
                        if new_arr_r2 is None:
                            continue

                    old_cost = route_cost(route1, f1, instance, M, E) + \
                               route_cost(route2, f2, instance, M, E)
                    new_cost = route_cost(new_r1, f1, instance, M, E) + \
                               route_cost(cand, f2, instance, M, E)

                    delta = old_cost - new_cost
                    
                    if delta > 0:
                        R[r1]["route"] = new_r1
                        if 'arrival' not in R[r1]:
                            R[r1]["arrival"] = new_arr_r1  # Update stored arrival times
                        else:
                            R[r1]["arrival"] = new_arr_r1
                        R[r2]["route"] = cand
                        if 'arrival' not in R[r2]:
                            R[r2]["arrival"] = new_arr_r2  # Update stored arrival times  
                        else:
                            R[r2]["arrival"] = new_arr_r2
                        return True

    # No improving move found
    return False

def relocate_all(R, instance, M, E):
    """
    Try relocating nodes until no better solution is found 
    """
    improvements = True
    while improvements:
        improvements = relocate_once(R, instance, M, E)
    return R


### Solution file formatting ### 

def export_routes_csv(R, path="routes.csv"):
    routes_list = []
    max_len = 0

    for r in sorted(R.keys()):
        fam = R[r]["family"]
        route = R[r]["route"]
        delivery_points = [node for node in route if node != 0]
        max_len = max(max_len, len(delivery_points))
        routes_list.append([fam] + delivery_points)

    df = pd.DataFrame(routes_list)
    df = df.apply(lambda col: col.fillna(""))
    df = df.map(lambda x: "" if x == "" else str(int(x)))
    df.columns = ["family"] + [f"order_{i}" for i in range(1, max_len + 1)]
    df.to_csv(path, index=False)


### Running the search on 10 different instances ###

M1, E1 = delta_M(instance1), delta_E(instance1)
R1 = build_solution(instance1, M1, E1)
R1 = relocate_all(R1, instance1, M1, E1)
export_routes_csv(R1, path="routes1.csv")
travel_cache.clear()

M2, E2 = delta_M(instance2), delta_E(instance2)
R2 = build_solution(instance2, M2, E2)
R2 = relocate_all(R2, instance2, M2, E2)
export_routes_csv(R2, path="routes2.csv")
travel_cache.clear()

M3, E3 = delta_M(instance3), delta_E(instance3)
R3 = build_solution(instance3, M3, E3)
R3 = relocate_all(R3, instance3, M3, E3)
export_routes_csv(R3, path="routes3.csv")
travel_cache.clear()

M4, E4 = delta_M(instance4), delta_E(instance4)
R4 = build_solution(instance4, M4, E4)
R4 = relocate_all(R4, instance4, M4, E4)
export_routes_csv(R4, path="routes4.csv")
travel_cache.clear()

M5, E5 = delta_M(instance5), delta_E(instance5)
R5 = build_solution(instance5, M5, E5)
R5 = relocate_all(R5, instance5, M5, E5)
export_routes_csv(R5, path="routes5.csv")
travel_cache.clear()

M6, E6 = delta_M(instance6), delta_E(instance6)
R6 = build_solution(instance6, M6, E6)
R6 = relocate_all(R6, instance6, M6, E6)
export_routes_csv(R6, path="routes6.csv")
travel_cache.clear()

M7, E7 = delta_M(instance7), delta_E(instance7)
R7 = build_solution(instance7, M7, E7)
R7 = relocate_all(R7, instance7, M7, E7)
export_routes_csv(R7, path="routes7.csv")
travel_cache.clear()

M8, E8 = delta_M(instance8), delta_E(instance8)
R8 = build_solution(instance8, M8, E8)
R8 = relocate_all(R8, instance8, M8, E8)
export_routes_csv(R8, path="routes8.csv")
travel_cache.clear()

M9, E9 = delta_M(instance9), delta_E(instance9)
R9 = build_solution(instance9, M9, E9)
R9 = relocate_all(R9, instance9, M9, E9)
export_routes_csv(R9, path="routes9.csv")
travel_cache.clear()

M10, E1O = delta_M(instance10), delta_E(instance10)
R10 = build_solution(instance10, M10, E10)
R10 = relocate_all(R10, instance10, M10, E10)
export_routes_csv(R10, path="routes10.csv")
travel_cache.clear()
