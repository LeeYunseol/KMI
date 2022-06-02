import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv('2018.csv')

data['PCI- West Coast North America with 4 lag'] = data['PCI- West Coast North America'].shift(4)
#data['PCI- West Coast North America with 4 lag'].dropna(axis=0, inplace=True)

#값이 너무 커서 전처리를 해줘야할듯함
 
data = data.iloc[:, 1:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)


from dtw import *

#feature = data_df_scaled['Average Earnings'].values
test_target = data_df_scaled['SCFI'].values
print('SCFI Sequence Length : ' + str(len(test_target)))
feature = data_df_scaled['PCI- West Coast North America with 4 lag'].dropna(axis=0)
print('PCI- West Coast North America with 4 lag Sequence Length : ' + str(len(feature)))
test_feature = feature.values


alignment = dtw(test_feature, test_target, keep_internals=True)
alignment.plot(type='threeway')
dtw(test_feature, test_target, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type='twoway', offset=-2)
    
print(dtw(test_feature, test_target, keep_internals= True).distance)