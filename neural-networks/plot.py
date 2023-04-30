import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob

files = glob.glob('./data/plots/*')
for f in files:
    os.remove(f)

files = glob.glob('./data/confusion/*')
for f in files:
    confusion = pd.read_csv("./data/confusion/relu-2-100.csv")
    confusion = confusion.reset_index(drop=True)
    confusion = confusion.iloc[:,1:]
    sns.heatmap(confusion,annot=True, fmt="d")
    plt.savefig("./data/plots/"+f[16:-4]+".png")
    plt.clf()


