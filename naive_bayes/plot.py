import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob

files = glob.glob('./data/naive-bayes/plots/*')
for f in files:
    os.remove(f)

confusion = pd.read_csv("./data/naive-bayes/confusion.csv")
confusion = confusion.reset_index(drop=True)
confusion = confusion.iloc[:,1:]
sns.heatmap(confusion,annot=True, fmt="d")
plt.savefig("./data/naive-bayes/confusion.png")
plt.clf()