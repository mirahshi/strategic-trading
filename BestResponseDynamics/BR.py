import numpy as np

def best_respond(V, bnow, T, kappa, lower_limit, upper_limit):
    """
    Compute best response from class of strategies that buy shares between [lower_limit, upper_limit] at every round.
    
    Parameters:
    -----------
    V : int
        Total shares to buy
    bnow : array-like
        Opponent's trading schedule (shares bought at each time step)
    T : int
        Number of time steps
    kappa : float
        Relative multiplier on permanent impact
    lower_limit : int
        Lower limit of shares that can be bought at each step
    upper_limit : int
        Upper limit of shares that can be bought at each step
        
    Returns:
    --------
    anow : array
        Optimal trading schedule (shares bought at each time step)
    """
    # cumulative shares held by opponent each time step
    b = np.array([np.sum(bnow[:i+1]) for i in range(T)])

    # helper functions to determine min/max number of remaining shares at each time step
    # can hold between [lower_limit*t, upper_limit*t] shares at time t, and so shares remaining are between [V-upper_limit*t, V-lower_limit*t] at time t
    def min_remaining(t):
        return V - upper_limit * t
    def max_remaining(t):
        return V - lower_limit * t

    # range of remaining shares
    s_size = max_remaining(T) - min_remaining(T) + 1

    dptable = np.full((T, s_size), np.inf) # initialize table of optimal costs
    dpaction = np.zeros((T, s_size))  # initialize table of optimal actions

    # helper function to map remaining shares s at time t to DP table index
    def s_to_index(s):
        return s - min_remaining(T)

    # initialization: on last step (t = T), must buy/sell all remaining shares
    for s in range(min_remaining(T), max_remaining(T) + 1):
        if lower_limit <= s <= upper_limit: # must adhere to trading limits
            i = s_to_index(s)
            dptable[T-1,i] = s*((s + bnow[T-1]) + kappa*((V-s) + b[T-2]))
            dpaction[T-1,i] = s

    # populate DP table from time step T-2 to 0
    for t in reversed(range(T-1)): 
        for s in range(min_remaining(t), max_remaining(t) + 1):  # shares remaining at time r
            i = s_to_index(s) # corresponding index in DP table

            # find optimal number of shares to trade now
            mincost = np.inf
            for q in range(lower_limit, upper_limit + 1): # possible shares to trade
                next_s = s - q # remaining shares after buying q shares

                #if min_remaining(t+1) <= next_s <= max_remaining(t+1):
                next_i = s_to_index(next_s) # corresponding index
                if t == 0:
                    # at first step, permanent impact is 0
                    thisval = dptable[t+1,next_i] + q*(q + bnow[t])
                else:
                    thisval = dptable[t+1,next_i] + q*((q + bnow[t]) + kappa*((V-s) + b[t-1])) # A holds V-s shares, B holds b[t-1] shares

                if thisval < mincost: # update opt q for this s
                    dptable[t,i] = thisval
                    dpaction[t,i] = q
                    mincost = thisval

    # compute optimal strategy for A from DP table
    anow = np.zeros(T) # a': trading schedule for A

    # traverse DP table, starting from time step 0 with V shares remaining
    volrem = V
    for t in range(T):
        i = s_to_index(int(volrem))
        anow[t] = dpaction[t,i]
        volrem -= anow[t]

    # return optimal strategy
    return anow

def total_cost(anow, bnow, kappa):
    """
    Calculate total cost of anow against bnow.
    
    Parameters:
    -----------
    anow : array-like
        Trading schedule for player A
    bnow : array-like
        Trading schedule for player B
    kappa : float
        Relative multiplier on permanent impact
        
    Returns:
    --------
    total_cost : float
        Total cost for player A
    """
    a = np.array([np.sum(anow[:i]) for i in range(len(anow))]) # A's cumulative trading schedule (0 at t=0)
    b = np.array([np.sum(bnow[:i]) for i in range(len(bnow))]) # B's cumulative trading schedule (0 at t=0)

    total_cost = 0
    for t in range(len(anow)):
        total_cost += ((anow[t] + bnow[t]) * anow[t]) + (kappa * (a[t] + b[t]) * anow[t])

    return total_cost
