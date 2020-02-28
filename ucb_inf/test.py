import pandas as pd

a = {'ite': [1,2,3,1,2,3],

'gp':['a','a','b','b','a','b'],
'val':[1,1,1,1,1,1]}


df = pd.DataFrame(a)

df['x'] = df.groupby(['ite','gp']).cumcount()+1

print(df)

