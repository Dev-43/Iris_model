import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data =load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

print(df.head())
print(df.describe)
print(df['target'].value_counts())

df.hist(figsize=(8,6))
plt.show()