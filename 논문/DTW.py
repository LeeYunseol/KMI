import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dtw import *

data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)
print(data.columns)
#%%

# 쉬프트하고 MinMaxScaler를 적용




columns = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']

new_columns = ['PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']

lag_list = [-8, -4, +4, +8]
for column in tqdm(columns) :
    for lag in lag_list :
        if (lag > 0) :
            new_column = column + ' with ' + "+" + str(lag) +' week lag'
        else :
            new_column = column + ' with ' + str(lag) +' week lag'
        new_columns.append(new_column)
        data[new_column] = data[column].shift(lag)
#new_columns.remove('SCFI')
new_columns.remove('SCFI with -4 week lag')
new_columns.remove('SCFI with +4 week lag')
new_columns.remove('SCFI with -8 week lag')
new_columns.remove('SCFI with +8 week lag')

# MinMaxScaler
data = data.iloc[:, 1:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)


target = data_df_scaled['SCFI'].values

box_plot_2018 =[]
print("========== 2018년 데이터 ==========")
for column in new_columns :
    temp = data_df_scaled[column].dropna(axis=0)
    feature = temp.values
    plt.title(column)
    alignment = dtw(feature, target, keep_internals=True)
    alignment.plot(type='threeway')
    dtw(feature, target, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type='twoway', offset=-2)
    box_plot_2018.append(dtw(feature, target, keep_internals= True).distance)
    print(column + '과의 시계적 유사도는 ' + str(round(dtw(feature, target, keep_internals= True).distance,15)))
    
green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(box_plot_2018, flierprops=green_diamond)

plt.show()

data_box_plot_2018 = pd.Series(box_plot_2018)
threshold_2018 = np.percentile(data_box_plot_2018, 20)
#%%
threshold_dtw_2 = np.percentile(data_box_plot_2018, 50)