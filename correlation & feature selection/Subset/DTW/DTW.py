import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv('2018.csv')
print(data)

#값이 너무 커서 전처리를 해줘야할듯함
 
data = data.iloc[:, 1:]
print(data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)


from dtw import *

#feature = data_df_scaled['Average Earnings'].values
target = data_df_scaled['SCFI'].values


#alignment = dtw(feature, target, keep_internals=True)
#alignment.plot(type='threeway')
#dtw(feature, target, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type='twoway', offset=-2)


#print(dtw(feature, target, keep_internals= True).distance)

columns = ['PCI-Comprehensive', 'PCI- East Coast North America',
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
box_plot_2018 =[]
print("========== 2018년 데이터 ==========")
for column in columns :
    feature = data_df_scaled[column].values
    plt.title(column)
    alignment = dtw(feature, target, keep_internals=True)
    alignment.plot(type='threeway')
    dtw(feature, target, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type='twoway', offset=-2)
    box_plot_2018.append(dtw(feature, target, keep_internals= True).distance)
    print(column + '과의 시계적 유사도는 ' + str(dtw(feature, target, keep_internals= True).distance))
    
green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(box_plot_2018, flierprops=green_diamond)
plt.title("DTW 2018 box plot")
plt.show()

data_box_plot_2018 = pd.Series(box_plot_2018)
threshold = np.percentile(data_box_plot_2018, 20)
#%%
# 2020년 데이터 DTW 분석

data = pd.read_csv('2020.csv')
print(data.columns)

#값이 너무 커서 전처리를 해줘야할듯함

data = data.iloc[:, 1:]
print(data.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)


target = data_df_scaled['SCFI'].values

columns = ['SCFI-Europe (base port)', 'SCFI-Med (base port)',
       'SCFI-WC America (base port)', 'SCFI-EC America (base port)',
       'SCFI-Persian Gulf (Dubai)', 'SCFI-ANZ (Melbourne)',
       'SCFI-W Africa (Lagos)', 'SCFI-S Africa (Durban)',
       'SCFI-S America (Santos)', 'SCFI-W Japan (base port)',
       'SCFI-E Japan (base port)', 'SCFI-SE Asia (Singapore)',
       'SCFI-Korea (Pusan)', 'PCI-Comprehensive',
       'PCI- East Coast North America', 'PCI- West Coast North America',
       'PCI- United Kingdom,Continent', 'PCI- Mediterranean,Black Sea',
       'PCI- East Asia', 'PCI- South East Asia', 'PCI- China P.R.',
       'Clarksons Average Containership Earnings',
       'HSFO 380cst Bunker Prices (3.5% Sulphur), Rotterdam',
       'Total Containerships - % Idle,Laid Up,Scrubber Retrofit',
       'Total Containerships - % Idle,Laid Up,Scrubber Retrofit.1',
       'Containership 1,650,1,850 TEU FCC, G\'less Newbuilding Prices',
       'Containership 13,000,14,000 TEU Newbuilding Prices',
       'Containership 3,500,4,000 TEU (Wide Beam) G\'less Newbuilding Prices',
       'Containership 13,000,13,500 TEU G\'less Newbuilding Prices',
       '5 Year $10m Finance based on Libor 1st yr']

box_plot_2020 =[]
print("========== 2020년 데이터 ==========")
for column in columns :
    feature = data_df_scaled[column].values
    plt.title(column)
    alignment = dtw(feature, target, keep_internals=True)
    alignment.plot(type='threeway')
    dtw(feature, target, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type='twoway', offset=-2)
    box_plot_2020.append(dtw(feature, target, keep_internals= True).distance)
    print(column + '과의 시계적 유사도는 ' + str(round((dtw(feature, target, keep_internals= True).distance),2)))
    
green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(box_plot_2020, flierprops=green_diamond)
plt.title("DTW 2020 box plot")
plt.show()

data_box_plot_2020 = pd.Series(box_plot_2020)
threshold = np.percentile(data_box_plot_2020, 20)
#%%
# 2018년 DTW를 기준으로 Subset 만들기
data = pd.read_csv('2018.csv')
subset_2018_dtw = data[['Date', 'SCFI','Average Earnings', 'Newbuilding Prices(3500,4000 TEU)',
                        'Newbuilding Prices(13000,13500 TEU)', 'PCI-Comprehensive']]
subset_2018_dtw.to_csv("subset_2018_dtw.csv", index = False)
#%%
# 2020년 DTW를 기준으로 Subset 만들기
data = pd.read_csv('2020.csv')
subset_2020_dtw = data[['Date', 'SCFI','Containership 13,000,13,500 TEU G\'less Newbuilding Prices',
                        'SCFI-Europe (base port)', 'SCFI-Med (base port)',
                        'SCFI-WC America (base port)', 'SCFI-EC America (base port)',
                        'SCFI-ANZ (Melbourne)']]
subset_2020_dtw.to_csv("subset_2020_dtw.csv", index = False)