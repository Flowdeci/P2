from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log
import random

num_nodes = 250
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



def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    MAX_DEPTH = 5  # Limit the depth of rollouts
    depth = 0
    
    # Play the game randomly until it ends or the depth limit is reached.
    while not board.is_ended(state) and depth < MAX_DEPTH:
        action = random.choice(board.legal_actions(state))
        state = board.next_state(state, action)
        depth += 1

    # Ensure the state is terminal before returning
    while not board.is_ended(state):
        action = random.choice(board.legal_actions(state))
        state = board.next_state(state, action)

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
