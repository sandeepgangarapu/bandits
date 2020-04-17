import pandas as pd

df = pd.read_csv("G:\My Drive\Projects\COVID\pincode_data.csv")

df_state = df.loc
a = {}
for i,j in zip(df.pincode, df.statename):
    a[i] = j

with open("pincode_state.json", 'w') as f:
    f.write(str(a))

print(df['pincode'].value_counts())
df = df.set_index('pincode')
print(df.head)
df.to_json("pin_dist_stat.json", orient='index')