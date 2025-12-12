import numpy as np
import matplotlib.pyplot as plt


def plot_winrate(wrs, title, warmup=0, out=None):
    wrs = wrs[warmup:]
    plt.plot(np.linspace(1, len(wrs), len(wrs)), wrs)
    plt.title(title)
    plt.xlabel("Games")
    plt.ylabel("Winrate")
    plt.tight_layout()
    if out:
        plt.savefig(out)
    else:
        plt.show()