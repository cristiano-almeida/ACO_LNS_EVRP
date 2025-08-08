import math
import random
import numpy as np
import sys
import os
import csv
import matplotlib.pyplot as plt

# --- PARSER E CLASSE DE INSTÂNCIA (sem alterações) ---
class Instance:
    def __init__(self):
        self.name, self.num_vehicles, self.capacity, self.energy_capacity, self.energy_consumption = "", 0, 0, 0, 0.0
        self.nodes, self.demands, self.stations = {}, {}, []
        self.depot_id, self.num_customers, self.num_stations = 0, 0, 0
        self.distance_matrix = None

def parse_evrp_instance(file_path):
    instance = Instance()
    with open(file_path, 'r') as f:
        lines = f.readlines()
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("COMMENT") or line == "EOF": continue
        if ":" in line:
            key, value = [x.strip() for x in line.split(':', 1)]
            if key == "NAME": instance.name = value
            elif key == "VEHICLES": instance.num_vehicles = int(value)
            elif key == "CAPACITY": instance.capacity = int(value)
            elif key == "ENERGY_CAPACITY": instance.energy_capacity = float(value)
            elif key == "ENERGY_CONSUMPTION": instance.energy_consumption = float(value)
            elif key == "STATIONS": instance.num_stations = int(value)
            elif key == "DIMENSION": instance.num_customers = int(value) - 1
            continue
        if line == "NODE_COORD_SECTION": section = "nodes"
        elif line == "DEMAND_SECTION": section = "demands"
        elif line == "STATIONS_COORD_SECTION": section = "stations"
        elif line == "DEPOT_SECTION": section = "depot"
        else:
            if section is None: continue
            parts = [p for p in line.split(' ') if p]
            if section == "nodes": instance.nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "demands": instance.demands[int(parts[0])] = int(parts[1])
            elif section == "stations": instance.stations.append(int(parts[0]))
            elif section == "depot":
                if int(parts[0]) != -1: instance.depot_id = int(parts[0])
    instance.demands[instance.depot_id] = 0
    for station_id in instance.stations: instance.demands[station_id] = 0
    max_id = max(instance.nodes.keys())
    instance.distance_matrix = np.zeros((max_id + 1, max_id + 1))
    for i in instance.nodes:
        for j in instance.nodes:
            if i == j: continue
            dist = math.hypot(instance.nodes[i][0] - instance.nodes[j][0], instance.nodes[i][1] - instance.nodes[j][1])
            instance.distance_matrix[i, j] = dist
    return instance

# --- SOLVER ACO + LNS (VERSÃO FINAL OTIMIZADA) ---
class ACO_LNS_Solver:
    def __init__(self, instance, **params):
        self.instance = instance
        self.num_ants, self.iterations, self.alpha, self.beta, self.rho = params.get("num_ants"), params.get("iterations"), params.get("alpha"), params.get("beta"), params.get("rho")
        self.q_param, self.lns_iter_limit = params.get("q_param"), params.get("lns_iter_limit")
        self.pheromone, self.heuristic, self.best_solution_overall, self.best_distance_overall = None, None, None, float('inf')
        self.convergence_curve = []

    def _initialize(self):
        max_id = max(self.instance.nodes.keys())
        self.heuristic = np.zeros((max_id + 1, max_id + 1))
        for i in range(1, max_id + 1):
            for j in range(1, max_id + 1):
                if i != j and self.instance.distance_matrix[i, j] > 0: self.heuristic[i, j] = 1.0 / self.instance.distance_matrix[i, j]
        self.pheromone = np.full((max_id + 1, max_id + 1), 1.0)
        _, initial_cost = self._construct_solution_for_ant(use_pheromone=False)
        if initial_cost == float('inf'): initial_cost = self.instance.num_customers * 100
        self.tau_max = 1.0 / (self.rho * initial_cost)
        self.tau_min = self.tau_max / (2 * self.instance.num_customers)
        self.pheromone = np.full((max_id + 1, max_id + 1), self.tau_max)
        self.best_solution_overall, self.best_distance_overall = None, float('inf')
        self.convergence_curve = []

    def solve(self):
        self._initialize()
        for it in range(self.iterations):
            best_iter_solution, best_iter_distance = None, float('inf')
            for _ in range(self.num_ants):
                solution, distance = self._construct_solution_for_ant()
                if distance < best_iter_distance:
                    best_iter_solution, best_iter_distance = solution, distance
            if best_iter_solution:
                best_iter_solution, best_iter_distance = self._large_neighborhood_search(best_iter_solution)
            if best_iter_distance < self.best_distance_overall:
                self.best_distance_overall, self.best_solution_overall = best_iter_distance, best_iter_solution
            self._update_pheromone()
            self.convergence_curve.append(self.best_distance_overall)
        return self.best_solution_overall, self.best_distance_overall

    def _large_neighborhood_search(self, initial_solution):
        current_sol, current_dist = [r[:] for r in initial_solution], self._calculate_solution_dist(initial_solution)
        for _ in range(self.lns_iter_limit):
            temp_sol, removed_customers = self._destroy_solution(current_sol)
            new_sol = self._repair_solution(temp_sol, removed_customers)
            new_dist = self._calculate_solution_dist(new_sol)
            if new_dist < current_dist:
                current_sol, current_dist = new_sol, new_dist
        return current_sol, current_dist

    def _destroy_solution(self, solution):
        destroyed_solution = [r[:] for r in solution]
        all_customers = [node for route in solution for node in route if self.instance.demands.get(node, 0) > 0]
        if not all_customers: return destroyed_solution, []
        num_to_remove = int(len(all_customers) * self.q_param)
        removed_customers = random.sample(all_customers, k=min(num_to_remove, len(all_customers)))
        for r_idx, route in enumerate(destroyed_solution):
            destroyed_solution[r_idx] = [node for node in route if node not in removed_customers]
        return [r for r in destroyed_solution if len(r) > 2], removed_customers

    def _repair_solution(self, partial_solution, customers_to_insert):
        repaired_solution = [r[:] for r in partial_solution]
        customers_to_insert = list(customers_to_insert)
        random.shuffle(customers_to_insert)
        
        for cust_id in customers_to_insert:
            best_insertion = {'cost_increase': float('inf'), 'route_idx': -1, 'pos': -1}
            for r_idx, route in enumerate(repaired_solution):
                for pos in range(1, len(route)):
                    cost_increase = self._get_insertion_cost(route, pos, cust_id)
                    if cost_increase < best_insertion['cost_increase']:
                        best_insertion = {'cost_increase': cost_increase, 'route_idx': r_idx, 'pos': pos}
            new_route_cost = self._get_new_route_cost(cust_id)
            if new_route_cost < best_insertion['cost_increase']:
                best_insertion = {'cost_increase': new_route_cost, 'route_idx': -1, 'pos': -1}
            if best_insertion['route_idx'] != -1:
                repaired_solution[best_insertion['route_idx']].insert(best_insertion['pos'], cust_id)
            elif best_insertion['cost_increase'] != float('inf'):
                repaired_solution.append([self.instance.depot_id, cust_id, self.instance.depot_id])
        
        # --- CORREÇÃO DE ROBUSTEZ FINAL ---
        # Valida e limpa a solução antes de retorná-la
        valid_solution = []
        all_served_customers = set()
        for route in repaired_solution:
            # Remove duplicações consecutivas (o bug original)
            cleaned_route = [route[0]] + [route[i] for i in range(1, len(route)) if route[i] != route[i-1]]
            if len(cleaned_route) > 2:
                valid_solution.append(cleaned_route)
                for node in cleaned_route:
                    if self.instance.demands.get(node, 0) > 0:
                        all_served_customers.add(node)
        
        # Se algum cliente foi perdido, a solução é inválida, mas para evitar travar,
        # simplesmente retornamos o que temos (será penalizado pelo _calculate_solution_dist)
        return valid_solution

    def _get_insertion_cost(self, route_nodes, pos, cust_id):
        current_load = sum(self.instance.demands.get(c,0) for c in route_nodes)
        if current_load + self.instance.demands[cust_id] > self.instance.capacity:
            return float('inf')
        
        hypothetical_route = route_nodes[:pos] + [cust_id] + route_nodes[pos:]
        if not self._is_route_energy_feasible(hypothetical_route):
            return float('inf')
        
        original_dist = self._calculate_route_dist(route_nodes)
        new_dist = self._calculate_route_dist(hypothetical_route)
        return new_dist - original_dist

    def _get_new_route_cost(self, cust_id):
        if self.instance.demands[cust_id] > self.instance.capacity: return float('inf')
        hypothetical_route = [self.instance.depot_id, cust_id, self.instance.depot_id]
        if self._is_route_energy_feasible(hypothetical_route):
            return self._calculate_route_dist(hypothetical_route)
        return float('inf')

    def _is_route_energy_feasible(self, route_nodes, try_to_repair=True):
        energy = self.instance.energy_capacity
        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i+1]
            energy_consumed = self.instance.distance_matrix[u, v] * self.instance.energy_consumption
            if energy < energy_consumed:
                if not try_to_repair: return False # Apenas checa, não repara
                # Lógica de reparo reativo da versão anterior, que é rápida
                best_station, dist_us, dist_sv = self._find_best_recharge_station(u, v, energy)
                if not best_station: return False
                # Não modifica a rota, apenas simula o que aconteceria
                energy_sim = self.instance.energy_capacity - (dist_sv * self.instance.energy_consumption)
                if energy_sim < 0: return False
                energy = energy_sim
            else:
                energy -= energy_consumed
            if v in self.instance.stations:
                energy = self.instance.energy_capacity
        return True
    
    # ... (Restante das funções auxiliares, como _construct_solution_for_ant, _find_best_recharge_station, etc.)
    def _construct_solution_for_ant(self, use_pheromone=True):
        solution, total_distance = [], 0.0
        customers_to_visit = [node_id for node_id, demand in self.instance.demands.items() if demand > 0]
        depot_id = self.instance.depot_id
        while customers_to_visit:
            current_route, current_dist = [depot_id], 0.0
            current_capacity, current_energy = self.instance.capacity, self.instance.energy_capacity
            last_node = depot_id
            while True:
                next_node, needs_recharge, station = self._select_next_node(last_node, customers_to_visit, current_capacity, current_energy, use_pheromone)
                if next_node is None: break
                if needs_recharge:
                    dist_to_station = self.instance.distance_matrix[last_node, station]
                    dist_from_station = self.instance.distance_matrix[station, next_node]
                    current_route.extend([station, next_node])
                    current_dist += dist_to_station + dist_from_station
                    current_energy = self.instance.energy_capacity - (dist_from_station * self.instance.energy_consumption)
                else:
                    dist_to_next = self.instance.distance_matrix[last_node, next_node]
                    current_route.append(next_node)
                    current_energy -= dist_to_next * self.instance.energy_consumption
                    current_dist += dist_to_next
                current_capacity -= self.instance.demands.get(next_node, 0)
                last_node = next_node
                customers_to_visit.remove(next_node)
            energy_to_depot = self.instance.distance_matrix[last_node, depot_id] * self.instance.energy_consumption
            if current_energy < energy_to_depot:
                best_final_station, dist_to_station, dist_from_station = self._find_best_recharge_station(last_node, depot_id, current_energy)
                if best_final_station:
                    current_route.append(best_final_station)
                    current_dist += dist_to_station + dist_from_station
                else: current_dist = float('inf') 
            else:
                 current_dist += self.instance.distance_matrix[last_node, depot_id]
            current_route.append(depot_id)
            solution.append(current_route)
            total_distance += current_dist
        return solution, total_distance

    def _select_next_node(self, current_node, candidates, current_capacity, current_energy, use_pheromone=True):
        possible_choices = []
        depot_id = self.instance.depot_id
        for node_id in candidates:
            if self.instance.demands.get(node_id, 0) > current_capacity: continue
            energy_to_next = self.instance.distance_matrix[current_node, node_id] * self.instance.energy_consumption
            energy_from_next_to_depot = self.instance.distance_matrix[node_id, depot_id] * self.instance.energy_consumption
            if current_energy >= energy_to_next + energy_from_next_to_depot:
                possible_choices.append({'node': node_id, 'recharge': False, 'station': None, 'cost': self.instance.distance_matrix[current_node, node_id]})
                continue
            station, dist_to_s, dist_from_s = self._find_best_recharge_station(current_node, node_id, current_energy)
            if station is not None:
                energy_from_s_to_depot = self.instance.distance_matrix[node_id, depot_id] * self.instance.energy_consumption
                if self.instance.energy_capacity >= (dist_from_s * self.instance.energy_consumption) + energy_from_s_to_depot:
                     possible_choices.append({'node': node_id, 'recharge': True, 'station': station, 'cost': dist_to_s + dist_from_s})
        if not possible_choices: return None, False, None
        probabilities, total_prob = [], 0.0
        for choice in possible_choices:
            tau = self.pheromone[current_node, choice['node']] ** self.alpha if use_pheromone else 1.0
            eta = (1.0 / choice['cost']) ** self.beta if choice['cost'] > 0 else 1.0
            prob = tau * eta
            probabilities.append(prob)
            total_prob += prob
        if total_prob == 0:
            choice = random.choice(possible_choices)
            return choice['node'], choice['recharge'], choice['station']
        r = random.uniform(0, total_prob)
        upto = 0
        for i, choice in enumerate(possible_choices):
            upto += probabilities[i]
            if upto >= r: return choice['node'], choice['recharge'], choice['station']
        choice = random.choice(possible_choices)
        return choice['node'], choice['recharge'], choice['station']

    def _find_best_recharge_station(self, from_node, to_node, current_energy):
        best_station, min_total_dist = None, float('inf')
        for station_id in self.instance.stations:
            dist_us = self.instance.distance_matrix[from_node, station_id]
            dist_sv = self.instance.distance_matrix[station_id, to_node]
            energy_to_station = dist_us * self.instance.energy_consumption
            energy_from_station = dist_sv * self.instance.energy_consumption
            if current_energy >= energy_to_station and self.instance.energy_capacity >= energy_from_station:
                if (dist_us + dist_sv) < min_total_dist:
                    min_total_dist = dist_us + dist_sv
                    best_station = station_id
        if best_station:
            return best_station, self.instance.distance_matrix[from_node, best_station], self.instance.distance_matrix[best_station, to_node]
        return None, None, None

    def _calculate_solution_dist(self, solution):
        if not solution: return float('inf')
        served_customers = {node for route in solution for node in route if self.instance.demands.get(node, 0) > 0}
        if len(served_customers) != self.instance.num_customers:
            return float('inf')
        return sum(self._calculate_route_dist(route) for route in solution)

    def _calculate_route_dist(self, route):
        return sum(self.instance.distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

    def _update_pheromone(self):
        self.pheromone *= (1 - self.rho)
        if self.best_solution_overall:
            deposit_amount = 1.0 / self.best_distance_overall
            for route in self.best_solution_overall:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i+1]
                    self.pheromone[u, v] += deposit_amount
                    self.pheromone[v, u] += deposit_amount
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

# --- FUNÇÕES DE VISUALIZAÇÃO E EXECUÇÃO ---
def plot_convergence(instance_name, convergence_curve, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_curve, marker='.', linestyle='-', markersize=4)
    plt.title(f"Curva de Convergência para {instance_name}\n(Melhor Execução)")
    plt.xlabel("Iteração"), plt.ylabel("Melhor Distância Encontrada")
    plt.grid(True), plt.tight_layout(), plt.savefig(filename), plt.close()

def plot_solution(instance, solution, distance, filename):
    plt.figure(figsize=(12, 12))
    depot_coord = instance.nodes[instance.depot_id]
    plt.plot(depot_coord[0], depot_coord[1], 'k*', markersize=15, label='Depósito')
    cust_coords = np.array([instance.nodes[i] for i in instance.nodes if instance.demands.get(i, 0) > 0])
    plt.plot(cust_coords[:, 0], cust_coords[:, 1], 'bo', markersize=5, label='Clientes')
    if instance.stations:
        station_coords = np.array([instance.nodes[i] for i in instance.stations])
        plt.plot(station_coords[:, 0], station_coords[:, 1], 'gs', markersize=8, label='Estações')
    colors = plt.cm.jet(np.linspace(0, 1, len(solution)))
    for i, route in enumerate(solution):
        if not route: continue
        route_coords = np.array([instance.nodes[node_id] for node_id in route])
        plt.plot(route_coords[:, 0], route_coords[:, 1], color=colors[i], linestyle='-', marker='o', markersize=3, alpha=0.8)
    plt.title(f"Melhor Rota Encontrada para {instance.name}\nDistância Total: {distance:.2f}")
    plt.xlabel("Coordenada X"), plt.ylabel("Coordenada Y")
    plt.legend(), plt.grid(True), plt.savefig(filename), plt.close()

def save_results_to_csv(instance_name, all_run_results, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Run', 'Distance', 'Num_Routes', 'Route']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run_idx, data in enumerate(all_run_results):
            solution = data.get('solution')
            if not solution: continue
            for route_idx, route in enumerate(solution):
                writer.writerow({'Run': run_idx + 1 if route_idx == 0 else '', 'Distance': f"{data['distance']:.2f}" if route_idx == 0 else '', 'Num_Routes': len(solution) if route_idx == 0 else '', 'Route': ' -> '.join(map(str, route))})

def main():
    instances_to_solve = {"E-n23-k3.evrp": 571.94, "E-n51-k5.evrp": 529.90}
    num_runs = 20
    aco_lns_params = {
        "num_ants": 5, "iterations": 100, "alpha": 1.0, "beta": 2.0, "rho": 0.6,
        "q_param": 0.35, "lns_iter_limit": 150
    }
    
    print("Iniciando simulações com solver ACO + LNS (versão final otimizada)...")
    
    for filename, known_optimum in instances_to_solve.items():
        print("\n" + "="*60), print(f"Resolvendo instância: {filename} (Ótimo: {known_optimum})"), print("="*60)
        if not os.path.exists(filename): print(f"ERRO: '{filename}' não encontrado."); continue
            
        instance_data = parse_evrp_instance(filename)
        run_results_values, all_run_results_details, all_convergence_curves = [], [], []
        
        for i in range(num_runs):
            sys.stdout.write(f"Execução {i+1}/{num_runs}... "), sys.stdout.flush()
            solver = ACO_LNS_Solver(instance_data, **aco_lns_params)
            best_sol, best_dist = solver.solve()
            run_results_values.append(best_dist)
            all_run_results_details.append({'solution': best_sol, 'distance': best_dist})
            all_convergence_curves.append(solver.convergence_curve)
            print(f"Resultado: {best_dist:.2f}")

        min_val, mean_val, std_dev, max_val = np.min(run_results_values), np.mean(run_results_values), np.std(run_results_values), np.max(run_results_values)
        print("\n--- Resultados Finais para " + filename + " ---")
        print(f"Mínimo:   {min_val:.2f}\nMédia:    {mean_val:.2f}\nDesv. Padrão: {std_dev:.2f}\nMáximo:   {max_val:.2f}")
        
        instance_basename = filename.split('.')[0]
        csv_filename = f"resultados_{instance_basename}_LNS_final.csv"
        plot_filename = f"melhor_rota_{instance_basename}_LNS_final.png"
        convergence_filename = f"convergencia_{instance_basename}_LNS_final.png"
        
        save_results_to_csv(instance_basename, all_run_results_details, csv_filename)
        print(f"\nResultados detalhados salvos em: {csv_filename}")
        
        best_run_idx = np.argmin(run_results_values)
        if all_run_results_details[best_run_idx].get('solution'):
            plot_solution(instance_data, all_run_results_details[best_run_idx]['solution'], all_run_results_details[best_run_idx]['distance'], plot_filename)
            print(f"Gráfico da melhor rota salvo em: {plot_filename}")
            plot_convergence(instance_basename, all_convergence_curves[best_run_idx], convergence_filename)
            print(f"Gráfico de convergência salvo em: {convergence_filename}")
        print("-"*(len(filename) + 24))

if __name__ == "__main__":
    main()