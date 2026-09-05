import pandas as pd, numpy as np
df = pd.read_csv("results/quartets.csv")   # adjust path
print(df.columns.tolist())                 # confirm names first

dA  = df.f_A  - df.f_WT
dB  = df.f_B  - df.f_WT
I   = df.f_A + df.f_B - df.f_WT - df.f_AB  # same sign convention as your epsilon

print("single-effect SD:", np.std(pd.concat([dA, dB])))
print("interaction SD:  ", np.std(I))
print("ratio:           ", np.std(I) / np.std(pd.concat([dA, dB])))
print(I.describe())
I.hist(bins=60)