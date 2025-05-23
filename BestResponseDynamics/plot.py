import numpy as np
import matplotlib.pyplot as plt

def plot_strategies_instantaneous(anow, bnow):
    """
    Plot instantaneous trading strategies for both players.
    
    Parameters:
    -----------
    anow : array-like
        Trading schedule for player A
    bnow : array-like
        Trading schedule for player B
    """
    T = len(anow)

    # Create the bar plot
    f, ax = plt.subplots(figsize=(5,3))
    plt.bar(np.arange(T) - 0.2, anow, width=0.4, label="a'", color='orange')
    plt.bar(np.arange(T) + 0.2, bnow, width=0.4, label="b'", color='cornflowerblue')

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Shares Bought')
    plt.title("Comparison of a' and b'")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_strategies_cumulative(anow, bnow):
    """
    Plot cumulative trading strategies for both players.
    
    Parameters:
    -----------
    anow : array-like
        Trading schedule for player A
    bnow : array-like
        Trading schedule for player B
    """
    T = len(anow)
    Va = sum(anow)
    Vb = sum(bnow)

    # recover shares held at each step
    a = np.cumsum(anow)
    b = np.cumsum(bnow)

    # Create the bar plot
    plt.bar(np.arange(T) - 0.2, a, width=0.4, label='a', color='orange')
    plt.bar(np.arange(T) + 0.2, b, width=0.4, label='b', color='cornflowerblue')

    # Plot target shares
    plt.axhline(Va, linestyle='dashed', color='gold', label='Va')
    plt.axhline(Vb, linestyle='dashed', color='lightblue', label='Vb')

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Shares Held')
    plt.title('Comparison of a and b')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show() 