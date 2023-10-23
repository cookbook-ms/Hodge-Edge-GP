import pickle 
import pandas as pd
import numpy as np


# load the data from csv and add column names
df = pd.read_csv('forex_2018/flow_FX_1538755200.csv', header=None)
df.columns = ['midpoint', 'ask','bid']
# b1 and b2
df_b1 = pd.read_csv('forex_2018/B1_FX_1538755200.csv', header=None)
df_b2t = pd.read_csv('forex_2018/B2T_FX_1538755200.csv', header=None)

# convert to numpy array
b1 = df_b1.values
b2t = df_b2t.values
b2 = b2t.T
exrates = df.values

n1 = b1.shape[1]
# random state
np.random.seed(0)
random_perm = np.random.permutation(np.arange(n1))
to_remove_ids = random_perm[:int(n1*0.3)]

exrates = np.delete(exrates, to_remove_ids, axis=0)
b1 = np.delete(b1, to_remove_ids, axis=1)
# check the number of nonzeros per column
np.count_nonzero(b1, axis=0)

# remove the rows of b2 corresponding to the removed columns of b1
b2 = np.delete(b2, to_remove_ids, axis=0)
# remove the columns of b2 with number of nonzeros less than 3
to_remove_ids = np.where(np.count_nonzero(b2, axis=0) < 3)[0]
b2 = np.delete(b2, to_remove_ids, axis=1)

L1 = b1.T@b1 + b2@b2.T
L1_down = b1.T@b1
L1_up = b2@b2.T

# save the data
with open('forex_2018/forex_2018.pkl', 'wb') as f:
    pickle.dump(((b1, b2), (L1, L1_down, L1_up) , exrates), f)
    
