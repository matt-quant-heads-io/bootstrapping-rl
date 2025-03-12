import random
import copy
import math
from typing import List, Dict, Tuple, Callable, Set, Any, Optional

# ----- PART 1: GAME STATE AND CONTEXT -----

class GameState:
    """Simple game state to test our evolving language and interpreter"""
    def __init__(self):
        self.player_x = 0
        self.player_y = 0
        self.player_health = 100
        self.enemies = [(random.randint(5, 15), random.randint(5, 15)) for _ in range(3)]
        self.score = 0
        self.items = {"health_potion": 2, "weapon": 1}
        self.turn = 0
    
    def __str__(self):
        return (f"Player: ({self.player_x}, {self.player_y}), Health: {self.player_health}, "
                f"Score: {self.score}, Turn: {self.turn}, "
                f"Enemies: {self.enemies}, Items: {self.items}")

# ----- PART 2: LANGUAGE SYNTAX REPRESENTATION -----

class SyntaxNode:
    """Base class for language syntax tree nodes"""
    def __init__(self, node_type: str):
        self.node_type = node_type
        self.complexity = 1
    
    def mutate(self, available_nodes: List['SyntaxNode'], mutation_rate: float = 0.2) -> 'SyntaxNode':
        """Base mutation method - overridden by subclasses"""
        return self
    
    def copy(self) -> 'SyntaxNode':
        """Create a deep copy of this node"""
        return copy.deepcopy(self)

class TerminalNode(SyntaxNode):
    """Terminal node representing variables or constants"""
    def __init__(self, value: Any, data_type: str):
        super().__init__("terminal")
        self.value = value
        self.data_type = data_type
        self.complexity = 1
    
    def mutate(self, available_nodes: List[SyntaxNode], mutation_rate: float = 0.2) -> SyntaxNode:
        # Terminal nodes can mutate their values or transform into a different terminal
        if random.random() < mutation_rate:
            # 50% chance to change value, 50% chance to change to different terminal
            if random.random() < 0.5:
                if self.data_type == "number":
                    # Mutate numeric value
                    self.value += random.randint(-5, 5)
                elif self.data_type == "string":
                    # Mutate string by adding/removing a character
                    options = ["player", "enemy", "health", "score", "item"]
                    self.value = random.choice(options)
            else:
                # Choose a different terminal from available nodes
                terminals = [n for n in available_nodes if n.node_type == "terminal"]
                if terminals:
                    return random.choice(terminals).copy()
        return self
    
    def __str__(self):
        if self.data_type == "string":
            return f'"{self.value}"'
        return str(self.value)

class OperationNode(SyntaxNode):
    """Node representing operations like add, move, attack, etc."""
    def __init__(self, operation: str, children: List[SyntaxNode], return_type: str):
        super().__init__("operation")
        self.operation = operation
        self.children = children
        self.return_type = return_type
        self.complexity = 1 + sum(child.complexity for child in children)
    
    def mutate(self, available_nodes: List[SyntaxNode], mutation_rate: float = 0.2) -> SyntaxNode:
        # Operations can mutate in several ways:
        # 1. Change the operation itself
        # 2. Mutate one of its children
        # 3. Replace with a different operation node
        # 4. Replace with a terminal node (simplification)
        
        if random.random() < mutation_rate:
            mutation_type = random.choice(["operation", "child", "replace", "simplify"])
            
            if mutation_type == "operation":
                # Change operation type if compatible
                ops = {"add": "subtract", "subtract": "add", "multiply": "divide", 
                       "move": "attack", "attack": "move", "use": "check"}
                if self.operation in ops:
                    self.operation = ops[self.operation]
                    
            elif mutation_type == "child" and self.children:
                # Mutate a random child
                child_idx = random.randrange(len(self.children))
                self.children[child_idx] = self.children[child_idx].mutate(available_nodes, mutation_rate)
                
            elif mutation_type == "replace":
                # Replace with different operation node
                operations = [n for n in available_nodes if n.node_type == "operation"]
                if operations:
                    return random.choice(operations).copy()
                    
            elif mutation_type == "simplify" and self.complexity > 3:
                # Simplify by replacing with a terminal
                terminals = [n for n in available_nodes if n.node_type == "terminal" 
                            and n.data_type == self.return_type]
                if terminals:
                    return random.choice(terminals).copy()
        
        # Recursively mutate children with reduced probability
        for i in range(len(self.children)):
            if random.random() < mutation_rate / 2:
                self.children[i] = self.children[i].mutate(available_nodes, mutation_rate / 2)
        
        # Recalculate complexity
        self.complexity = 1 + sum(child.complexity for child in self.children)
        return self
    
    def __str__(self):
        return f"{self.operation}({', '.join(str(child) for child in self.children)})"

class RuleNode(SyntaxNode):
    """Node representing a game rule with condition and action"""
    def __init__(self, condition: SyntaxNode, action: SyntaxNode):
        super().__init__("rule")
        self.condition = condition
        self.action = action
        self.complexity = condition.complexity + action.complexity + 1
    
    def mutate(self, available_nodes: List[SyntaxNode], mutation_rate: float = 0.2) -> SyntaxNode:
        # Rules can mutate their condition or action
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                self.condition = self.condition.mutate(available_nodes, mutation_rate)
            else:
                self.action = self.action.mutate(available_nodes, mutation_rate)
        return self
    
    def __str__(self):
        return f"IF {self.condition} THEN {self.action}"

# ----- PART 3: INTERPRETER REPRESENTATION -----

class InterpreterFunction:
    """Represents a function in our evolving interpreter"""
    def __init__(self, name: str, implementation: Callable, arg_types: List[str], return_type: str):
        self.name = name
        self.implementation = implementation
        self.arg_types = arg_types
        self.return_type = return_type
        # Measure fitness based on how often this function is used in rules
        self.usage_count = 0
        # Complexity score for this function
        self.complexity = len(arg_types) + 1
    
    def execute(self, args: List[Any], game_state: GameState) -> Any:
        """Execute this function with the given arguments"""
        self.usage_count += 1
        return self.implementation(args, game_state)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'InterpreterFunction':
        """Mutate this interpreter function"""
        if random.random() < mutation_rate:
            # Clone the function and make modifications
            new_func = copy.deepcopy(self)
            
            # Possible mutations:
            # 1. Add a small random modifier to numeric operations
            # 2. Modify the behavior slightly
            # 3. Add a side effect
            
            if self.name in ["add", "subtract", "multiply", "divide"]:
                # Add a modifier to math operations
                original_impl = self.implementation
                modifier = random.uniform(0.8, 1.2)  # Scale result by 80%-120%
                
                def modified_impl(args, game_state):
                    result = original_impl(args, game_state)
                    if isinstance(result, (int, float)):
                        return result * modifier
                    return result
                
                new_func.implementation = modified_impl
            
            elif self.name in ["move", "attack"]:
                # Add side effect to actions
                original_impl = self.implementation
                
                def side_effect_impl(args, game_state):
                    result = original_impl(args, game_state)
                    # 50% chance to affect score, 50% chance to affect health
                    if random.random() < 0.5:
                        game_state.score += random.randint(-2, 5)
                    else:
                        game_state.player_health += random.randint(-2, 2)
                    return result
                
                new_func.implementation = side_effect_impl
            
            return new_func
        return self

class Interpreter:
    """The evolving interpreter that executes language constructs"""
    def __init__(self):
        # Initialize with basic functions
        self.functions: Dict[str, InterpreterFunction] = {}
        self._initialize_base_functions()
        
        # Track performance metrics
        self.execution_time = 0
        self.execution_count = 0
        self.errors = 0
    
    def _initialize_base_functions(self):
        """Set up the initial set of interpreter functions"""
        # Math operations
        self.functions["add"] = InterpreterFunction(
            "add",
            lambda args, _: args[0] + args[1],
            ["number", "number"],
            "number"
        )
        
        self.functions["subtract"] = InterpreterFunction(
            "subtract",
            lambda args, _: args[0] - args[1],
            ["number", "number"],
            "number"
        )
        
        # Game-specific operations
        self.functions["move"] = InterpreterFunction(
            "move",
            lambda args, game_state: self._move_player(args[0], args[1], game_state),
            ["number", "number"],
            "boolean"
        )
        
        self.functions["attack"] = InterpreterFunction(
            "attack",
            lambda args, game_state: self._attack_enemy(args[0], game_state),
            ["number"],
            "boolean"
        )
        
        self.functions["check_proximity"] = InterpreterFunction(
            "check_proximity",
            lambda args, game_state: self._check_enemy_proximity(args[0], game_state),
            ["number"],
            "boolean"
        )
        
        self.functions["use_item"] = InterpreterFunction(
            "use_item",
            lambda args, game_state: self._use_item(args[0], game_state),
            ["string"],
            "boolean"
        )
    
    def _move_player(self, dx: int, dy: int, game_state: GameState) -> bool:
        """Move player by dx, dy if possible"""
        game_state.player_x += dx
        game_state.player_y += dy
        # Ensure player stays in bounds (simple 20x20 grid)
        game_state.player_x = max(0, min(20, game_state.player_x))
        game_state.player_y = max(0, min(20, game_state.player_y))
        return True
    
    def _attack_enemy(self, enemy_idx: int, game_state: GameState) -> bool:
        """Attack enemy at the given index if in range"""
        if enemy_idx < 0 or enemy_idx >= len(game_state.enemies):
            return False
            
        enemy_x, enemy_y = game_state.enemies[enemy_idx]
        distance = math.sqrt((game_state.player_x - enemy_x)**2 + (game_state.player_y - enemy_y)**2)
        
        # Can only attack if close enough
        if distance <= 3:
            # Remove enemy and increase score
            game_state.enemies.pop(enemy_idx)
            game_state.score += 10
            return True
        return False
    
    def _check_enemy_proximity(self, threshold: float, game_state: GameState) -> bool:
        """Check if any enemy is within the threshold distance"""
        for enemy_x, enemy_y in game_state.enemies:
            distance = math.sqrt((game_state.player_x - enemy_x)**2 + (game_state.player_y - enemy_y)**2)
            if distance <= threshold:
                return True
        return False
    
    def _use_item(self, item_name: str, game_state: GameState) -> bool:
        """Use an item from inventory if available"""
        if item_name in game_state.items and game_state.items[item_name] > 0:
            game_state.items[item_name] -= 1
            
            # Apply item effects
            if item_name == "health_potion":
                game_state.player_health += 20
            elif item_name == "weapon":
                # Weapon increases attack range temporarily
                pass
                
            return True
        return False
    
    def execute(self, node: SyntaxNode, game_state: GameState) -> Any:
        """Execute a syntax node in the current game state"""
        try:
            self.execution_count += 1
            
            if node.node_type == "terminal":
                if node.data_type == "number":
                    return node.value
                elif node.data_type == "string":
                    return node.value
                    
            elif node.node_type == "operation":
                # Execute children first
                arg_values = [self.execute(child, game_state) for child in node.children]
                
                # Execute the operation
                if node.operation in self.functions:
                    return self.functions[node.operation].execute(arg_values, game_state)
                else:
                    self.errors += 1
                    return None
                    
            elif node.node_type == "rule":
                # Evaluate condition
                condition_result = self.execute(node.condition, game_state)
                
                # If condition is true, execute action
                if condition_result:
                    return self.execute(node.action, game_state)
                    
            return None
            
        except Exception as e:
            self.errors += 1
            return None
    
    def mutate(self, mutation_rate: float = 0.1) -> 'Interpreter':
        """Create a mutated copy of this interpreter"""
        new_interpreter = copy.deepcopy(self)
        
        # Mutate existing functions
        for func_name in new_interpreter.functions:
            if random.random() < mutation_rate:
                new_interpreter.functions[func_name] = new_interpreter.functions[func_name].mutate(mutation_rate)
        
        # Possibly add a new function by combining existing ones
        if random.random() < mutation_rate * 0.3 and len(self.functions) >= 2:
            self._add_composite_function(new_interpreter)
        
        return new_interpreter
    
    def _add_composite_function(self, interpreter: 'Interpreter'):
        """Create a new function by composing existing ones"""
        # Select two random functions to compose
        funcs = list(interpreter.functions.values())
        func1, func2 = random.sample(funcs, 2)
        
        # Only compose if output of func1 matches input of func2
        if func1.return_type == func2.arg_types[0]:
            new_name = f"{func1.name}_{func2.name}"
            
            def composite_implementation(args, game_state):
                # Execute first function
                intermediate = func1.execute(args, game_state)
                # Feed result to second function
                return func2.execute([intermediate], game_state)
            
            new_function = InterpreterFunction(
                new_name,
                composite_implementation,
                func1.arg_types,
                func2.return_type
            )
            
            interpreter.functions[new_name] = new_function

# ----- PART 4: EVOLUTIONARY ALGORITHM -----

class Language:
    """Represents a language with its syntax and available constructs"""
    def __init__(self):
        # Initialize with basic syntax components
        self.available_nodes: List[SyntaxNode] = []
        self._initialize_basic_nodes()
        
        # Track language metrics
        self.expressiveness_score = 0
        self.consistency_score = 0
    
    def _initialize_basic_nodes(self):
        """Set up the initial available syntax nodes"""
        # Terminal nodes
        self.available_nodes.extend([
            TerminalNode(1, "number"),
            TerminalNode(5, "number"),
            TerminalNode(10, "number"),
            TerminalNode("health_potion", "string"),
            TerminalNode("weapon", "string")
        ])
        
        # Basic operation nodes
        self.available_nodes.extend([
            OperationNode("add", [TerminalNode(1, "number"), TerminalNode(2, "number")], "number"),
            OperationNode("move", [TerminalNode(1, "number"), TerminalNode(0, "number")], "boolean"),
            OperationNode("attack", [TerminalNode(0, "number")], "boolean"),
            OperationNode("check_proximity", [TerminalNode(3, "number")], "boolean"),
            OperationNode("use_item", [TerminalNode("health_potion", "string")], "boolean")
        ])
    
    def generate_random_rule(self) -> RuleNode:
        """Generate a random rule using available nodes"""
        # Create condition (mostly check_proximity)
        condition_ops = [n for n in self.available_nodes 
                        if n.node_type == "operation" and n.return_type == "boolean"]
        condition = random.choice(condition_ops).copy()
        
        # Create action (move, attack, or use_item)
        action_ops = [n for n in self.available_nodes 
                     if n.node_type == "operation"]
        action = random.choice(action_ops).copy()
        
        return RuleNode(condition, action)
    
    def mutate(self, mutation_rate: float = 0.2) -> 'Language':
        """Create a mutated copy of this language"""
        new_language = copy.deepcopy(self)
        
        # Mutate existing nodes
        for i in range(len(new_language.available_nodes)):
            if random.random() < mutation_rate:
                new_language.available_nodes[i] = new_language.available_nodes[i].mutate(
                    new_language.available_nodes, mutation_rate)
        
        # Possibly add a new node by combining existing ones
        if random.random() < mutation_rate * 0.5:
            self._add_composite_node(new_language)
        
        return new_language
    
    def _add_composite_node(self, language: 'Language'):
        """Create a new node by composing existing ones"""
        operations = [n for n in language.available_nodes if n.node_type == "operation"]
        terminals = [n for n in language.available_nodes if n.node_type == "terminal"]
        
        if operations and terminals:
            # Choose a random operation and replace its children with random terminals
            operation = random.choice(operations).copy()
            for i in range(len(operation.children)):
                compatible_terminals = [t for t in terminals 
                                      if hasattr(t, 'data_type') and t.data_type == operation.children[i].data_type]
                if compatible_terminals:
                    operation.children[i] = random.choice(compatible_terminals).copy()
            
            # Add the new composite node to available nodes
            language.available_nodes.append(operation)

class RuleSet:
    """A set of game rules expressed in the language"""
    def __init__(self, rules: List[RuleNode]):
        self.rules = rules
        self.fitness = 0
    
    def mutate(self, language: Language, mutation_rate: float = 0.3) -> 'RuleSet':
        """Create a mutated copy of this rule set"""
        new_ruleset = copy.deepcopy(self)
        
        # Mutate existing rules
        for i in range(len(new_ruleset.rules)):
            if random.random() < mutation_rate:
                new_ruleset.rules[i] = new_ruleset.rules[i].mutate(language.available_nodes, mutation_rate)
        
        # Possibly add or remove a rule
        if random.random() < mutation_rate:
            if random.random() < 0.7 and len(new_ruleset.rules) < 10:
                # Add a new rule
                new_ruleset.rules.append(language.generate_random_rule())
            elif len(new_ruleset.rules) > 1:
                # Remove a rule
                del new_ruleset.rules[random.randrange(len(new_ruleset.rules))]
        
        return new_ruleset
    
    def crossover(self, other: 'RuleSet') -> 'RuleSet':
        """Create a new rule set by combining rules from this and another rule set"""
        # Choose a random crossover point for each parent
        crossover_point1 = random.randrange(len(self.rules) + 1)
        crossover_point2 = random.randrange(len(other.rules) + 1)
        
        # Create new rule set with rules from both parents
        new_rules = self.rules[:crossover_point1] + other.rules[crossover_point2:]
        
        return RuleSet(new_rules)
    
    def __str__(self):
        return "\n".join([f"Rule {i}: {rule}" for i, rule in enumerate(self.rules)])

# ----- PART 5: CO-EVOLUTION ALGORITHM -----

def evaluate_fitness(ruleset: RuleSet, language: Language, interpreter: Interpreter) -> Tuple[float, float, float]:
    """Evaluate the fitness of a ruleset, language, and interpreter combination"""
    game_state = GameState()
    
    # Reset interpreter stats
    interpreter.execution_count = 0
    interpreter.errors = 0
    
    # Play through a simple game simulation
    for _ in range(20):  # 20 turns
        game_state.turn += 1
        
        # Execute each rule
        for rule in ruleset.rules:
            interpreter.execute(rule, game_state)
            
        # Simulate basic game logic
        if random.random() < 0.2:  # 20% chance each turn
            # Add a new enemy
            game_state.enemies.append((random.randint(5, 15), random.randint(5, 15)))
    
    # Calculate ruleset fitness based on game outcome
    ruleset_fitness = (
        game_state.score +                                # Higher score is better
        game_state.player_health +                        # Higher health is better
        (20 - len(game_state.enemies)) * 5                # Fewer enemies is better
    )
    
    # Language fitness based on expressiveness and usage
    used_operations = set()
    for rule in ruleset.rules:
        def collect_operations(node):
            if node.node_type == "operation":
                used_operations.add(node.operation)
                for child in node.children:
                    collect_operations(child)
            elif node.node_type == "rule":
                collect_operations(node.condition)
                collect_operations(node.action)
        
        collect_operations(rule)
    
    language_fitness = (
        len(used_operations) * 10 +                       # More diverse operations is better
        len(language.available_nodes)                     # More available nodes is better
    )
    
    # Interpreter fitness based on execution statistics
    interpreter_fitness = (
        interpreter.execution_count * 2 -                 # More execution is good
        interpreter.errors * 10 +                         # Errors are bad
        len(interpreter.functions) * 5                    # More functions is good
    )
    
    return ruleset_fitness, language_fitness, interpreter_fitness

def co_evolution(generations: int = 20, population_size: int = 10):
    """Run the co-evolutionary algorithm"""
    # Initialize populations
    languages = [Language() for _ in range(population_size)]
    interpreters = [Interpreter() for _ in range(population_size)]
    
    # For each language/interpreter pair, create a rule set
    rulesets = []
    for i in range(population_size):
        rules = [languages[i].generate_random_rule() for _ in range(3)]
        rulesets.append(RuleSet(rules))
    
    # Initialize fitness tracking
    fitness_history = []
    
    # Main evolutionary loop
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations}")
        
        # Evaluate fitness of all combinations
        fitness_scores = []
        for i in range(population_size):
            ruleset_fitness, language_fitness, interpreter_fitness = evaluate_fitness(
                rulesets[i], languages[i], interpreters[i])
            
            # Store fitness values
            rulesets[i].fitness = ruleset_fitness
            languages[i].expressiveness_score = language_fitness
            interpreters[i].execution_time = interpreter_fitness
            
            total_fitness = ruleset_fitness + language_fitness + interpreter_fitness
            fitness_scores.append((i, total_fitness))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = fitness_scores[0][0]
        
        # Print best combination
        print(f"Best fitness: {fitness_scores[0][1]}")
        print(f"Best ruleset:\n{rulesets[best_idx]}")
        print(f"Language nodes: {len(languages[best_idx].available_nodes)}")
        print(f"Interpreter functions: {len(interpreters[best_idx].functions)}")
        print("-" * 40)
        
        # Store best fitness
        fitness_history.append(fitness_scores[0][1])
        
        # Create next generation
        new_rulesets = []
        new_languages = []
        new_interpreters = []
        
        # Elitism: keep the best 2 unchanged
        for i in range(2):
            if i < len(fitness_scores):
                idx = fitness_scores[i][0]
                new_rulesets.append(copy.deepcopy(rulesets[idx]))
                new_languages.append(copy.deepcopy(languages[idx]))
                new_interpreters.append(copy.deepcopy(interpreters[idx]))
        
        # Fill the rest with mutations and crossovers
        while len(new_rulesets) < population_size:
            # Select parents based on fitness (tournament selection)
            parent_indices = random.sample(range(population_size), 4)
            parent_indices.sort(key=lambda i: fitness_scores[i][1], reverse=True)
            parent1_idx, parent2_idx = parent_indices[:2]
            
            # Create child through crossover of rulesets
            child_ruleset = rulesets[parent1_idx].crossover(rulesets[parent2_idx])
            
            # Create child language through mutation of better parent's language
            child_language = languages[parent1_idx].mutate(mutation_rate=0.2)
            
            # Create child interpreter through mutation of better parent's interpreter
            child_interpreter = interpreters[parent1_idx].mutate(mutation_rate=0.1)
            
            # Mutate the child ruleset
            child_ruleset = child_ruleset.mutate(child_language, mutation_rate=0.3)
            
            # Add to new generation
            new_rulesets.append(child_ruleset)
            new_languages.append(child_language)
            new_interpreters.append(child_interpreter)
        
        # Replace old generation
        rulesets = new_rulesets
        languages = new_languages
        interpreters = new_interpreters
    
    # Return the best evolved system
    best_idx = max(range(population_size), key=lambda i: rulesets[i].fitness)
    return rulesets[best_idx], languages[best_idx], interpreters[best_idx], fitness_history

# ----- PART 6: DEMONSTRATION -----

def demonstrate_evolved_system():
    """Run the co-evolution algorithm and demonstrate the results"""
    print("Starting co-evolution of language, interpreter, and rules...")
    best_ruleset, best_language, best_interpreter, fitness_history = co_evolution(generations=10, population_size=10)
    
    print("\nEvolution complete!")
    print(f"Final fitness: {best_ruleset.fitness}")
    print("\nEvolved rule set:")
    print(best_ruleset)
    
    print("\nAvailable language constructs:")
    for i, node in enumerate(best_language.available_nodes):
        print(f"Node {i}: {node}")
    
    print("\nAvailable interpreter functions:")
    for name, func in best_interpreter.functions.items():
        print(f"Function: {name}, Args: {func.arg_types}, Return: {func.return_type}, Usage: {func.usage_count}")
    
    print("\nRunning a game simulation with the evolved system...")
    game_state = GameState()
    
    print("Initial state:")
    print(game_state)
    
    for turn in range(10):
        print(f"\nTurn {turn+1}:")
        # Execute each rule
        for i, rule in enumerate(best_ruleset.rules):
            result = best_interpreter.execute(rule, game_state)
            print(f"Rule {i} executed with result: {result}")
        
        # Display current state
        print(game_state)
    
    print("\nCo-evolution demonstration complete!")

if __name__ == "__main__":
    demonstrate_evolved_system()