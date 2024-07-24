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
    marker_weights = {
        "delta_credit": 76.58059571136468,
        "delta_battery": 7.341291533674699,
        "delta_pack": 5.534079750825349,
        "credit": 93.74017271498576,
        "battery": 27.416059349426625,
        "pack_bonus": 82.46689438553943,
        "distance_to_target": 13.87751731938508,  # This marker has NEGATIVE value
        "distance_to_charger": 0.9502993298955875  # This marker has NEGATIVE value
    }

    # Calculate markers that give the robot advantage
    if other_robot.battery == 0 and robot.credit-other_robot.credit > 0:
        return float("inf")
    markers = {}
    markers["delta_credit"] = robot.credit - other_robot.credit
    markers["credit"] = robot.credit
    markers["delta_battery"] = robot.battery - other_robot.battery
    markers["battery"] = robot.battery
    has_pack = 1 if robot.package else 0
    rival_has_pack = 1 if other_robot.package else 0
    markers["delta_pack"] = has_pack - rival_has_pack
    markers["pack_bonus"] = has_pack

    if not robot.package:
        closest_pack = -min([manhattan_distance(robot.position, pack.position) for pack in env.packages if pack.on_board])
    markers["distance_to_target"] = closest_pack if not robot.package else -manhattan_distance(robot.position, robot.package.destination)

    closest_charger = min([manhattan_distance(robot.position, charger.position) for charger in env.charge_stations])
    markers["distance_to_charger"] = -(closest_charger+1) * charger_gain

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
        self.DNA = DNA(8, 0, 100)
        self.genetic_worth = 0
        self.wins = 0

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id, self.DNA)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.best_move = None
        self.original = agent_id
        iterations = 1
        try:
            func_timeout.func_timeout(time_limit-0.2, self.anytime_step, args=(env, self.original, iterations))
        except func_timeout.FunctionTimedOut:
            return self.best_move

    def anytime_step(self, env: WarehouseEnv, agent_id, iterations):
        operators = env.get_legal_operators(agent_id)
        children = self.apply_moves(agent_id, env)
        while True:
            child_values = [self.value(child, agent_id, iterations) for child in children]
            self.best_move = operators[child_values.index(max(child_values))]
            iterations += 1

    def value(self, state: WarehouseEnv, agent_id, iterations):
        if state.done():
            return state.get_robot(self.original).credit - state.get_robot((self.original + 1) % 2).credit
        if iterations == 0:
            return smart_heuristic(state, self.original, None)
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
        self.trajectory = ["move west", "move west", "move south"]

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
