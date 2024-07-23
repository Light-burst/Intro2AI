from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import func_timeout
# TODO: section a : 3


def smart_heuristic(env: WarehouseEnv, robot_id: int, dna):
    # Get details about the robots
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    bat_percent = robot.battery / 20
    charger_gain = (min(20, bat_percent + robot.credit) - bat_percent)

    # Weight of each marker
    # marker_weights = {
    #     "delta_credit": dna.features[0],
    #     "delta_battery": dna.features[1],
    #     "delta_pack": dna.features[2],
    #     "pack_bonus": dna.features[3],
    #     # This marker has NEGATIVE value
    #     "distance_to_target": dna.features[4],
    #     # This marker has NEGATIVE value
    #     "distance_to_charger": dna.features[5]
    # }
    marker_weights = {
        "delta_credit": 20,
        "delta_battery": 1.154619400439284,
        "delta_pack": 6.358148000063247,
        "pack_bonus": 5.515369624976212,
        "distance_to_target": 1.2415100626400688,  # This marker has NEGATIVE value
        "distance_to_charger": 5.972927491904429  # This marker has NEGATIVE value
    }

    # Calculate markers that give the robot advantage
    # TODO: seperate deltas from stand alone
    markers = {}
    markers["delta_credit"] = robot.credit - other_robot.credit
    markers["delta_battery"] = robot.battery - other_robot.battery
    has_pack = 1 if robot.package else 0
    rival_has_pack = 1 if other_robot.package else 0
    markers["delta_pack"] = has_pack - rival_has_pack
    markers["pack_bonus"] = has_pack
    if not robot.package:
        closest_pack = -min([manhattan_distance(robot.position, pack.position) for pack in env.packages])
        markers["distance_to_target"] = closest_pack
    else:
        markers["distance_to_target"] = -manhattan_distance(robot.position, robot.package.destination)
    markers["distance_to_target"] = bat_percent * markers["distance_to_target"]
    closest_charger = min([manhattan_distance(robot.position, charger.position) for charger in env.charge_stations])
    markers["distance_to_charger"] = -closest_charger * charger_gain #TODO: add flat debuff

    return sum([markers[key] * marker_weights[key] for key in markers.keys()])


class DNA:
    def __init__(self, num_features, feature_min, feature_max, mutation_chance=0.1):
        self.features = []
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.mutation_chance = mutation_chance
        for feature in range(num_features):
            self.features.append(random.random() * (feature_max-feature_min))

    def __str__(self):
        result = ""
        for i, feature in enumerate(self.features):
            result += str(i)+": "+str(self.features[i]) + "\n"
        return result

    def crossover(self, other):
        transposed = DNA(len(self.features),
                         self.feature_min, self.feature_max)
        for feature in range(len(self.features)):
            random_weight = random.random()
            transposed.features[feature] = random_weight*(self.features[feature]) + \
                (1-random_weight)*(other.features[feature])
            if random.random() < self.mutation_chance:
                transposed.features[feature] = random.random() * (self.feature_max-self.feature_min)
        return transposed


class AgentGreedyImproved(AgentGreedy):
    def __init__(self):
        super().__init__()
        self.DNA = DNA(6, 0, 10)
        self.genetic_worth = 0
        self.wins = 0
        # TODO: Add num generations, num mutations

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id, self.DNA)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.best_move = None
        self.original = agent_id
        iterations = 1
        try:
            func_timeout.func_timeout(time_limit-0.1, self.anytime_step, args=(env, self.original, iterations))
        except func_timeout.FunctionTimedOut:
            return self.best_move

    def anytime_step(self, env: WarehouseEnv, agent_id, iterations):
        operators = env.get_legal_operators(agent_id)
        children = self.apply_moves(agent_id, env)
        while True:
            child_values = [self.value(child, agent_id, iterations) for child in children]
            self.best_move = operators[child_values.index(max(child_values))]
            iterations += 1
            print(iterations)


    def value(self, state: WarehouseEnv, agent_id, iterations):
        if iterations>7:
            1+1
        if state.done():
            return state.get_robot(self.original).credit - state.get_robot((self.original + 1) % 2).credit
        if iterations == 0:
            return state.get_robot(self.original).credit - state.get_robot((self.original + 1) % 2).credit
            #return smart_heuristic(state, self.original, None)
        if agent_id == self.original:
            return self.max_value(state, agent_id, iterations)
        else:
            return self.min_value(state, agent_id, iterations)

    def max_value(self, state: WarehouseEnv, agent_id, iterations):
        new_agent_id = (agent_id + 1) % 2
        children = self.apply_moves(agent_id, state)
        return max([self.value(child, new_agent_id, iterations - 1) for child in children])

    def min_value(self, state: WarehouseEnv, agent_id, iterations):
        new_agent_id = (agent_id + 1) % 2
        children = self.apply_moves(agent_id, state)
        return min([self.value(child, new_agent_id, iterations - 1) for child in children])

    def apply_moves(self, agent, env: WarehouseEnv):
        operators = env.get_legal_operators(agent)
        children = [env.clone() for op in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent, op)
        return children


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move west", "move west", "move south", "pick up", "move east", "move east", "move east",
                           "drop off", "move east", "move south", "pick up", "move west", "move west", "move west",
                           "move south", "move south", "drop off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
