import pandas as pd
import glob

indices = [
    'model',
    'Loss',
    'Accuracy',
    'Precision',
    'Recall',
]

df = pd.DataFrame(columns=indices)


files = glob.glob('./data/metrics/*')
for f in files:
    tempdf = pd.read_csv(f,header=None, index_col=0)
    df.loc[len(df.index)]=[f[15:-4],tempdf.iloc[0][1],tempdf.iloc[1][1],tempdf.iloc[2][1],tempdf.iloc[3][1]]
df.to_csv('data/metrics.csv')