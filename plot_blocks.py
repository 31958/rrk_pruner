import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [('down_blocks[2].resnets[0]', 0.0), ('down_blocks[0].attentions[1]', 14.0), ('up_blocks[1].attentions[0]', 16.0),
 ('up_blocks[3].resnets[2]', 16.0), ('down_blocks[0].resnets[1]', 22.0), ('up_blocks[0].resnets[0]', 22.0),
 ('up_blocks[1].resnets[0]', 22.0), ('up_blocks[1].attentions[2]', 22.0), ('down_blocks[1].resnets[0]', 28.0),
 ('up_blocks[1].resnets[2]', 28.0), ('up_blocks[2].attentions[1]', 28.0), ('down_blocks[3].resnets[1]', 32.0),
 ('mid_block.attentions[0]', 32.0), ('down_blocks[0].attentions[0]', 36.0), ('down_blocks[1].attentions[0]', 36.0),
 ('down_blocks[1].resnets[1]', 36.0), ('down_blocks[1].attentions[1]', 36.0), ('up_blocks[3].resnets[0]', 36.0),
 ('down_blocks[2].attentions[1]', 40.0), ('up_blocks[3].attentions[0]', 44.0), ('down_blocks[0].resnets[0]', 46.0),
 ('down_blocks[3].resnets[0]', 46.0), ('up_blocks[3].resnets[1]', 46.0), ('mid_block.resnets[0]', 52.0),
 ('up_blocks[3].attentions[1]', 60.0), ('up_blocks[1].resnets[1]', 70.0), ('mid_block.resnets[1]', 76.0),
 ('up_blocks[2].resnets[2]', 88.0), ('up_blocks[1].attentions[1]', 94.0), ('up_blocks[2].resnets[1]', 102.0),
 ('up_blocks[0].resnets[1]', 126.0), ('up_blocks[2].attentions[2]', 130.0), ('down_blocks[2].resnets[1]', 138.0),
 ('up_blocks[2].attentions[0]', 140.0), ('up_blocks[2].resnets[0]', 150.0), ('down_blocks[2].attentions[0]', 162.0),
 ('up_blocks[0].resnets[2]', 166.0)]


# convert to DataFrame
df = pd.DataFrame(data, columns=['Layer', 'Value'])
df = df.sort_values("Layer")

# plot
plt.figure(figsize=(12, 10))
sns.barplot(data=df, x="Value", y="Layer", palette="viridis", order=df['Layer'])
plt.title("Layer Values Bar Chart (Original Order)")
plt.xlabel("Value")
plt.ylabel("Layer")
plt.tight_layout()
plt.show()