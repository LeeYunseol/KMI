# 이미지가 아닌 시계열 값만을 사용해서 코사인 유사도 비교하기 
import numpy as np
import pandas as pd
data = pd.read_csv('2018.csv')


#값이 너무 커서 전처리를 해줘야할듯함
 
#data = data.iloc[:, 1:]

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(data)
#data_scaled = scaler.transform(data)
#data_df_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)

#target = data_df_scaled['SCFI'].values
#feature = data_df_scaled['Average Earnings'].values
#feature2 = data_df_scaled['Total Container ships Number'].values
#print(cosine_similarity(feature, target))
#print(cosine_similarity(feature2, target))


# 이게 되네?
#%%
columns = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom,Continent',
       'PCI- Mediterranean,Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650,1850 TEU)',
       'Newbuilding Prices(13000,14000 TEU)',
       'Newbuilding Prices(3500,4000 TEU)',
       'Newbuilding Prices(13000,13500 TEU)', '5 Year Finance based on Libor',
       'Exchange Rates South Korea', 'Exchange Rates Euro',
       'Exchange Rates China']

new_columns = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom,Continent',
       'PCI- Mediterranean,Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650,1850 TEU)',
       'Newbuilding Prices(13000,14000 TEU)',
       'Newbuilding Prices(3500,4000 TEU)',
       'Newbuilding Prices(13000,13500 TEU)', '5 Year Finance based on Libor',
       'Exchange Rates South Korea', 'Exchange Rates Euro',
       'Exchange Rates China']

# lag는 8주 전, 4주 전, 4주 후, 4주 후(★이거는 내 맘대로 해서 피드백 필요)
lag_list = [-8, -4, +4, +8]
for column in tqdm(columns) :
    for lag in lag_list :
        if (lag > 0) :
            new_column = column + ' with ' + "+" + str(lag) +' week lag'
        else :
            new_column = column + ' with ' + str(lag) +' week lag'
        new_columns.append(new_column)  
        data[new_column] = data[column].shift(lag)
        
lag_after_4_col = []
lag_after_8_col = []
lag_before_8_col = []
lag_before_4_col = []

for column in data.columns:
    if '+4 week lag' in column:
        lag_after_4_col.append(column)
        
    elif '+8 week lag' in column :
        lag_after_8_col.append(column)
        
    elif'-8 week lag' in column : 
        lag_before_8_col.append(column)
        
    elif '-4 week lag' in column :
       lag_before_4_col.append(column)
       
data_original = pd.read_csv('2018.csv')
data_after_4_lag = data[lag_after_4_col]
data_after_8_lag = data[lag_after_8_col]
data_before_4_lag = data[lag_before_4_col]
data_before_8_lag = data[lag_before_8_col]

# garbage index 제거
data_after_4_lag.dropna(axis=0, inplace=True)
data_after_4_lag.reset_index(drop=True, inplace=True)

data_after_8_lag.dropna(axis=0, inplace=True)
data_after_8_lag.reset_index(drop=True,inplace=True)

data_before_4_lag.dropna(axis=0, inplace=True)
data_before_4_lag.reset_index(drop=True,inplace=True)

data_before_8_lag.dropna(axis=0, inplace=True)
data_before_4_lag.reset_index(drop=True,inplace=True)
#%%
def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    
    return similarity
#%%
#Original data 코사인 유사도 분석

data_original = data_original.iloc[:, 1:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_original)
data_scaled = scaler.transform(data_original)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data_original.columns)

target = data_df_scaled['SCFI'].values
print("\n========== Original Data ==========")
for column in data_original.columns:
    feature = data_df_scaled[column]
    cos_sim = cos_similarity(feature, target)
    print(column + "와 SCFI의 코사인 유사도는 " + str(round(cos_sim, 2)))
    
#%%
# +4 Week lag 데이터에 대한 코사인 유사도 분석
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_after_4_lag)
data_scaled = scaler.transform(data_after_4_lag)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data_after_4_lag.columns)

target = data_df_scaled['SCFI with +4 week lag'].values
print("\n========== with +4 lag ==========")
for column in data_after_4_lag.columns:
    feature = data_df_scaled[column]
    cos_sim = cos_similarity(feature, target)
    print(column + "와 SCFI의 코사인 유사도는 " + str(round(cos_sim, 2)))
#%%
# +8 Week lag 데이터에 대한 코사인 유사도 분석
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_after_8_lag)
data_scaled = scaler.transform(data_after_8_lag)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data_after_8_lag.columns)

target = data_df_scaled['SCFI with +8 week lag'].values
print("\n========== with +8 lag ==========")
for column in data_after_8_lag.columns:
    feature = data_df_scaled[column]
    cos_sim = cos_similarity(feature, target)
    print(column + "와 SCFI의 코사인 유사도는 " + str(round(cos_sim, 2)))
#%%
# -4 Week lag 데이터에 대한 코사인 유사도 분석
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_before_4_lag)
data_scaled = scaler.transform(data_before_4_lag)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data_before_4_lag.columns)

target = data_df_scaled['SCFI with -4 week lag'].values
print("\n========== with -4 lag ==========")
for column in data_before_4_lag.columns:
    feature = data_df_scaled[column]
    cos_sim = cos_similarity(feature, target)
    print(column + "와 SCFI의 코사인 유사도는 " + str(round(cos_sim, 2)))
#%%
# -8 Week lag 데이터에 대한 코사인 유사도 분석
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_before_8_lag)
data_scaled = scaler.transform(data_before_8_lag)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data_before_8_lag.columns)

target = data_df_scaled['SCFI with -8 week lag'].values
print("\n========== with -8 lag ==========")
for column in data_before_8_lag.columns:
    feature = data_df_scaled[column]
    cos_sim = cos_similarity(feature, target)
    print(column + "와 SCFI의 코사인 유사도는 " + str(round(cos_sim, 2)))
    