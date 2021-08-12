import pandas as pd
import matplotlib.pyplot as plt

rewards = pd.read_pickle("rewards.pkl")
plt.plot(range(len(rewards)), rewards)
plt.plot([0, 400], [195, 195], "--", color="darkred")
plt.xlabel("episodes")
plt.ylabel("Total Reward")
plt.savefig("rewards_fig.jpg")