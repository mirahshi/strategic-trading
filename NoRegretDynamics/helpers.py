import numpy as np
from collections import defaultdict, Counter
import sys
sys.path.append('..')
from BestResponseDynamics.BR import best_respond, total_cost


def regret(cumulative_cost, opponent_actions, V, T, kappa, lower_limit, upper_limit):
    """
    Computes cumulative external regret

    Args:
        cumulative_cost (float): Total cumulative cost
        opponent_actions (list): List of opponent actions
        V (float): Target trading volume
        T (float): Number of time steps
        kappa (float): Market impact parameter
        lower_limit (float): Minimum number of shares that can be traded in one time step
        upper_limit (float): Maximum number of shares that can be traded in one time step

    Returns:
        float: Cumulative (external) regret
    """
    r = len(opponent_actions)
    
    # compute sum of opponent actions
    bnow_r = np.add.reduce(opponent_actions) / r

    # compute best fixed action by best responding to sum of opponent actions
    best_action = best_respond(V, bnow_r, T, kappa, lower_limit, upper_limit)

    # compute regret
    best_cost = r * total_cost(best_action, bnow_r, kappa)

    return cumulative_cost - best_cost


def swap_regret(p1_actions, p2_actions, Va, T, kappa, lower_limit, upper_limit):
    """
    Computes cumulative swap regret of P1 against P2

    Args:
        p1_actions (list): List of Player 1's actions
        p2_actions (list): List of Player 2's actions
        Va (float): Target trading volume for Player 1
        T (float): Number of time steps
        kappa (float): Market impact parameter
        lower_limit (float): Minimum number of shares that can be traded in one time step
        upper_limit (float): Maximum number of shares that can be traded in one time step

    Returns:
        float: Cumulative swap regret
    """

    n = len(p1_actions)

    # Compute costs
    p1_costs = []
    for t in range(n):
        p1_costs.append(total_cost(p1_actions[t], p2_actions[t], kappa))

    # Convert to tuples for hashing
    p1_actions = [tuple(a) for a in p1_actions]
    p2_actions = [tuple(a) for a in p2_actions]

    # Group indices where Player 1 plays a specific action
    action_to_indices = defaultdict(list)
    for t, a1 in enumerate(p1_actions):
        action_to_indices[a1].append(t)

    total_regret = 0

    for a1, indices in action_to_indices.items():
        # Get the subsequence of Player 2's actions where Player 1 played a1
        p2_subseq = [p2_actions[t] for t in indices]
        
        subseq_cost = np.sum([p1_costs[t] for t in indices])
        total_regret += regret(subseq_cost, p2_subseq, Va, T, kappa, lower_limit, upper_limit)

    return total_regret


def marginal_cost(p1_actions, p2_actions, kappa):
    """
    Computes expected cost of Player 1's marginal distribution against Player 2's marginal distribution.

    Args:
        p1_actions (list): List of Player 1's actions
        p2_actions (list): List of Player 2's actions 
        kappa (float): Market impact parameter

    Returns:
        float: Expected cost under the product of marginal distributions
    """
    num_rounds = len(p1_actions)

    # construct marginal distributions over p1 and p2 actions
    p1_frequencies = Counter([tuple(p1_action) for p1_action in p1_actions])
    p1_dist = {action: freq / num_rounds for action, freq in p1_frequencies.items()}
    p2_frequencies = Counter([tuple(p2_action) for p2_action in p2_actions])
    p2_dist = {action: freq / num_rounds for action, freq in p2_frequencies.items()}
    
    cost = 0
    for a1, freq1 in p1_dist.items():
        for a2, freq2 in p2_dist.items():
            weight = freq1 * freq2
            cost += total_cost(a1, a2, kappa) * weight
    
    return cost


def dist_to_nash(p1_actions, p2_actions, Va, T, kappa, lower_limit, upper_limit):
    """
    Computes distance to Nash equilibrium

    Args:
        p1_actions (list): List of Player 1's actions
        p2_actions (list): List of Player 2's actions
        Va (float): Target trading volume
        T (float): Number of time steps
        kappa (float): Market impact parameter
        lower_limit (float): Minimum number of shares that can be traded in one time step
        upper_limit (float): Maximum number of shares that can be traded in one time step

    Returns:
        float: Distance to Nash equilibrium
    """
    actual_cost = marginal_cost(p1_actions, p2_actions, kappa)

    # compute average strategy of P2
    p2_avg_action = [sum(elements) / len(p2_actions) for elements in zip(*p2_actions)]

    # find best response against average strategy of P2
    p1_br = best_respond(Va, p2_avg_action, T, kappa, lower_limit, upper_limit)
    best_cost = total_cost(p1_br, p2_avg_action, kappa)

    return actual_cost - best_cost
  

def welfare(joint_dist, kappa):
    """
    Computes social welfare for a joint distribution of actions.

    Args:
        joint_dist (dict): Dictionary mapping tuples of actions (a,b) to their empirical probabilities
        kappa (float): Market impact parameter

    Returns:
        float: Expected total cost (welfare) summed across both players
    """
    welfare = 0

    # compute expected cost of both players
    for (a, b), prob in joint_dist.items():
        cost1 = total_cost(a, b, kappa)
        cost2 = total_cost(b, a, kappa)
        welfare += prob * (cost1 + cost2)
    
    return welfare