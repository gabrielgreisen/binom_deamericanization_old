import math

def binomial_crr(Option: str, K: int, T: int, S_0: float, sigma: float, r: float, q: float, N: int, Exercise: str):
    
    """
    Approximates the price of an option using the Cox-Ross-Rubinstein model

    Parameters
    -------------
    Option: str
        "C" for call options and "P" for put options
    K: int
        Strike price
    T: int
        Time to maturity
    S_0: float
        Initial stock price
    sigma: float
        Volatility
    r: float
        Continuous compounding risk free interest rate
    q: float
        Continuous dividend yield
    N: int
        Number of time steps
    Exercise: str
        "A" for American options and "E" for European options
    """


    # create timestep
    deltaT = T / N
    # factor that the instrument will move up by
    # down factor = 1 / up factor
    up = math.exp((sigma * math.sqrt(deltaT)))
    # probabilities of up and down moves
    p_up = (up * math.exp(-q * deltaT) - math.exp(-r * deltaT)) / (up ** 2 - 1)
    p_down = math.exp(-r * deltaT) - p_up
    tree_values = [0] * N
    # populating final tree values
    for i in range(N):
        # S_n = S_0 * u^{number of up ticks - number of down ticks}
        S_n = S_0 * up ** (2 * i - N + 1)
        # calculate exercise value at end
        tree_values[i] = (K - S_n) if Option == "P" else (S_n - K)
        # don't exercise if lose money
        tree_values[i] = max(tree_values[i], 0)
    # calculate earlier values
    for j in range(N - 1, -1, -1):
        for i in range(j):
            # S_i value = p * S_(i+1) up + (1 - p) * S_(i+1) down
            tree_values[i] = p_up * tree_values[i + 1] + p_down * tree_values[i]
            # American options can be exercised at any time
            if Exercise == "A":
                exercise_val = K - S_0 * up ** (2 * i - j)
                tree_values[i] = max(tree_values[i], exercise_val)
    # return approximated initial option value
    return tree_values[0]