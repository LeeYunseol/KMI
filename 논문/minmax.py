import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 3개의 방법론 중에서 3번 모두 공통적으로 나온 것 
data = pd.read_csv('subset_50.csv')
print(data.columns)
#%%
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Total Container ships Number' , 'Newbuilding Prices(13000/14000 TEU)',
                        'PCI-Comprehensive', 'PCI- West Coast North America'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_50.csv", index = False)

#%%

# 3개의 방법론 중에서 2번 공통적으로 나온 것

data = pd.read_csv('subset_2_of_3_selected.csv')
print(data.columns)
data_scaler = data.iloc[:, 1:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI', 'Average Earnings',
       'Newbuilding Prices(13000/13500 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)'])

data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_2_of_3_selected.csv", index = False)

#%%

# 비교대상 subset 만들기
# only mse
data = pd.read_csv('subset_only_mse.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
       'PCI- West Coast North America', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_only_mse.csv", index = False)
#%%

# 비교대상 subset 만들기
# ONLY CORR
data = pd.read_csv('subset_only_corr.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Average Earnings', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Newbuilding Prices(13000/14000 TEU)','5 Year Finance based on Libor','PCI- East Coast North America',
                        'PCI- West Coast North America','PCI- United Kingdom/Continent','PCI- South East Asia'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_only_corr.csv", index = False)
#%%

# 비교대상 subset 만들기
# ONLY DTW
data = pd.read_csv('subset_only_dtw.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'Average Earnings',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_only_dtw.csv", index = False)
#%%

# 비교대상 subset 만들기
# ONLY IMW2VEC
data = pd.read_csv('subset_only_img2vec.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_only_img2vec.csv", index = False)
#%%

#  subset 만들기
# 
data = pd.read_csv('last_subset.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI', 'PCI-Comprehensive', 'PCI- West Coast North America', 
                           'Average Earnings',
        'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_last_subset.csv", index = False)
#%%
# 20% 비교대상 subset 만들기
# MSE
data = pd.read_csv('subset_20_only_mse.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
        'Average Earnings',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_20_only_mse.csv", index = False)
#%%
# 20% 비교대상 subset 만들기
# DTW
data = pd.read_csv('subset_20_only_dtw.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI', 'PCI-Comprehensive', 'Average Earnings',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_20_only_dtw.csv", index = False)
#%%
# 20% 비교대상 subset 만들기
# IMG2VEC
data = pd.read_csv('subset_20_only_img2vec.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
       'PCI- West Coast North America', 'Average Earnings',
        'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_20_only_img2vec.csv", index = False)
#%%
# 20% 비교대상 subset 만들기
# Corr
data = pd.read_csv('subset_20_only_corr.csv')
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Average Earnings', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Newbuilding Prices(13000/14000 TEU)','PCI- East Coast North America',
                        'PCI- West Coast North America','PCI- United Kingdom/Continent'])



data_df_scaled['Original SCFI'] = data['SCFI']

data_df_scaled.to_csv("minmax_subset_20_only_corr.csv", index = False)