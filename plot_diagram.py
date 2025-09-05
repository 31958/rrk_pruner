import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("statistics.csv")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("statistics.csv")

sns.set(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis: SSIM + CLIP
ax1.plot(df["ratio"], df["ssim"], marker="o", color="red", label="SSIM")
ax1.plot(df["ratio"], df["clip"], marker="o", color="green", label="CLIP")
ax1.set_xlabel("Pruning Ratio")
ax1.set_ylabel("SSIM / CLIP")

# Right axis: FID
ax2 = ax1.twinx()
ax2.plot(df["ratio"], df["fid"], marker="o", color="blue", label="FID")
ax2.set_ylabel("FID", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

# Extend y-axis limits slightly so legend fits
ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1] * 1.2)
ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 1.2)

# Single legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="best")

plt.title("SSIM, CLIP, and FID vs Pruning Ratio")
plt.show()