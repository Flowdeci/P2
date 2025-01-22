
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 100
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    while not board.is_ended(state):
        # If the node has untried actions stop at this leaf node
        if node.untried_actions:
            return node, state

        # Otherwise select the best child node using UCB
        best_child = max(
            node.child_nodes.values(),
            key=lambda child: ucb(child, is_opponent=(bot_identity != board.current_player(state)))
        )

        # select the best child node
        node = best_child
        # Update the state to the next state based off the parent action
        state = board.next_state(state, node.parent_action)

    #if a terminal node is reached, return the node and state
    return node, state

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
   # Get an untried action from the node.
    action = node.untried_actions.pop()

    # Get the new state after taking the action.
    new_state = board.next_state(state, action)

    # Create a new child node for the selected action.
    child_node = MCTSNode(
        parent=node, # set the parent node, which is the current node
        parent_action=action, # get the action that led to this node
        action_list=board.legal_actions(new_state)  # Get all legal actions for the new state.
    )

    # Add the new child node to the parent node, and set the action that led to the child node.
    node.child_nodes[action] = child_node

    return child_node, new_state

def has_won_subboard(state, action, player):
    # Extract the subboard location and the player's current positions in the subboard.
    R, C, r, c = action
    subboard_index = 2 * (3 * R + C) + (player - 1)  # Player-specific subboard index.
    player_positions = state[subboard_index]

    # Define the win conditions (rows, columns, diagonals) for a 3x3 subboard.
    win_conditions = [
        0b111000000,  # Row 1
        0b000111000,  # Row 2
        0b000000111,  # Row 3
        0b100100100,  # Column 1
        0b010010010,  # Column 2
        0b001001001,  # Column 3
        0b100010001,  # Diagonal 1
        0b001010100,  # Diagonal 2
    ]

    # Check if the player's positions satisfy any win condition.
    return any(player_positions & condition == condition for condition in win_conditions)


def has_won_game(state, player):
    # Get the player's positions on the meta-board.
    player_boards = state[18 + (player - 1)]

    # Define the win conditions for the meta-board (3x3 grid of subboards).
    win_conditions = [
        0b111000000,  # Row 1
        0b000111000,  # Row 2
        0b000000111,  # Row 3
        0b100100100,  # Column 1
        0b010010010,  # Column 2
        0b001001001,  # Column 3
        0b100010001,  # Diagonal 1
        0b001010100,  # Diagonal 2
    ]

    # Check if the player's positions satisfy any win condition.
    return any(player_boards & condition == condition for condition in win_conditions)


def is_winning_move(board: Board, state, action):
    # Simulate the next state after applying the action.
    next_state = board.next_state(state, action)

    # Get the player whose turn it is.
    player = board.current_player(state)

    # Check if the player has won the subboard or the game.
    return has_won_subboard(next_state, action, player) or has_won_game(next_state, player)

def is_blocking_move(board: Board, state, action):
    # Simulate the next state after applying the action.
    next_state = board.next_state(state, action)

    # Get the opponent's player number.
    opponent = 3 - board.current_player(state)

    # Check if applying this action prevents the opponent from winning.
    return any(
        has_won_subboard(next_state, opp_action, opponent)
        for opp_action in board.legal_actions(next_state)
    )

def heuristic_action(board: Board, state, legal_actions):
    for action in legal_actions:
        # Simulate the next state for this action.
        state= board.next_state(state, action)
        
        # Check if this action wins the subboard for the current player.
        if is_winning_move(board, state, action):
            return action
        
        # Check if this action blocks the opponent from winning.
        if is_blocking_move(board, state, action):
            return action
    
    # If no critical moves are found, pick a random action (fallback).
    return choice(legal_actions)


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    
    # Play the game randomly until it ends.
    while not board.is_ended(state):
        legal_moves = board.legal_actions(state)  # Get all legal actions.
        action = heuristic_action(board, state, legal_moves)  # Select the best action using heuristics.
        
        # Apply the action to get the next state.
        state = board.next_state(state, action)

    # Return the terminal state.
    return state


def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    
    # Traverse up the tree until the root node is reached.
    while node is not None:
        # Increment the visit count for every node that lead to the leaf/terminal node.
        node.visits += 1
        # if the bot won, increment the win count for every node that lead to the leaf/terminal node.
        if won:
            node.wins += 1
        # Move up to the parent node.
        node = node.parent

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    #Nodes that havent been visted yet need to be explored asap so they are given a "high value".
    if node.visits == 0:
        return float('inf')  

    # Calcuate the explotiation rate and exploration rate
    exploration = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    exploitation = node.wins / node.visits
    
    # Adjust exploitation depending on whether it's the opponent's turn.
    return (1 - exploitation if is_opponent else exploitation) + exploration

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    
    return max(
        # Get the action with the highest number of visits
        # Nodes with higher visit counts have been explored more thoroughly, 
        # Meaning the algorithm has more data about their outcomes.
        root_node.child_nodes.keys(),
        key=lambda action: root_node.child_nodes[action].visits
    )

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS - This is all you!
        # ...

        # Selection: Traverse the tree to find a leaf node.
        node, state = traverse_nodes(node, board, state, bot_identity)

        # Expansion: Expand the leaf node if it has untried actions.
        if node.untried_actions:
            node, state = expand_leaf(node, board, state)

        # Simulation: Simulate a random game from the new state.
        terminal_state = rollout(board, state)

        # Backpropagation: Update the tree with the result of the simulation.
        won = is_win(board, terminal_state, bot_identity)
        backpropagate(node, won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
