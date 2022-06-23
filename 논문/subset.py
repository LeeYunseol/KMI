
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Exchanges rate 관련 3개의 변수는 사용하지 않겠음 => 총 17개 
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)
print(data.columns)
#%%

# 3 of 3 Selected
subset_3_of_3_selected = data[['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Total Container ships Number' , 'Newbuilding Prices(13000/14000 TEU)',
                        'PCI-Comprehensive', 'PCI- West Coast North America']]

# LAG 적용
subset_3_of_3_selected['Newbuilding Prices(13000/13500 TEU)'] = subset_3_of_3_selected['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_3_of_3_selected['Newbuilding Prices(3500/4000 TEU)'] = subset_3_of_3_selected['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
subset_3_of_3_selected['Total Container ships Number'] = subset_3_of_3_selected['Total Container ships Number'].shift(+8)
subset_3_of_3_selected['PCI- West Coast North America'] = subset_3_of_3_selected['PCI- West Coast North America'].shift(+8)
subset_3_of_3_selected['Newbuilding Prices(13000/14000 TEU)'] = subset_3_of_3_selected['Newbuilding Prices(13000/14000 TEU)'].shift(-8)
subset_3_of_3_selected['PCI-Comprehensive'] = subset_3_of_3_selected['PCI-Comprehensive'].shift(+8)
subset_3_of_3_selected.dropna(axis=0, inplace=True)
subset_3_of_3_selected.reset_index(drop=True,inplace=True)
subset_3_of_3_selected.to_csv("subset_50.csv", index = False)


#%%
# 비교대상 subset 만들기
# only mse

data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_only_mse = data[['SCFI',
       'PCI- West Coast North America', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']]


subset_only_mse['Average Earnings'] = subset_only_mse['Average Earnings'].shift(-4)
subset_only_mse['PCI- West Coast North America'] = subset_only_mse['PCI- West Coast North America'].shift(+8)
subset_only_mse['Bunker Prices'] = subset_only_mse['Bunker Prices'].shift(+8)
subset_only_mse['Total Container ships Number'] = subset_only_mse['Total Container ships Number'].shift(+8)
subset_only_mse['Total Container ships TEU'] = subset_only_mse['Total Container ships TEU'].shift(+8)
subset_only_mse['Newbuilding Prices(1650/1850 TEU)'] = subset_only_mse['Newbuilding Prices(1650/1850 TEU)'].shift(+8)
subset_only_mse['5 Year Finance based on Libor'] = subset_only_mse['5 Year Finance based on Libor'].shift(-8)
subset_only_mse['Newbuilding Prices(13000/13500 TEU)'] = subset_only_mse['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_only_mse['Newbuilding Prices(3500/4000 TEU)'] = subset_only_mse['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
subset_only_mse['Newbuilding Prices(13000/14000 TEU)'] = subset_only_mse['Newbuilding Prices(13000/14000 TEU)'].shift(-8)

subset_only_mse.dropna(axis=0, inplace=True)
subset_only_mse.reset_index(drop=True,inplace=True)
subset_only_mse.to_csv("subset_only_mse.csv", index = False)

#%%
# ONLY CORR
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_only_corr = data[['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Average Earnings', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Newbuilding Prices(13000/14000 TEU)','5 Year Finance based on Libor','PCI- East Coast North America',
                        'PCI- West Coast North America','PCI- United Kingdom/Continent','PCI- South East Asia']]                        

subset_only_corr['Newbuilding Prices(13000/13500 TEU)'] = subset_only_corr['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_only_corr['Average Earnings'] = subset_only_corr['Average Earnings'].shift(-8)
subset_only_corr['Newbuilding Prices(3500/4000 TEU)'] = subset_only_corr['Newbuilding Prices(3500/4000 TEU)'].shift(-8)
subset_only_corr['Newbuilding Prices(13000/14000 TEU)'] = subset_only_corr['Newbuilding Prices(13000/14000 TEU)'].shift(-8)
subset_only_corr['5 Year Finance based on Libor'] = subset_only_corr['5 Year Finance based on Libor'].shift(+8)
subset_only_corr['PCI- East Coast North America'] = subset_only_corr['PCI- East Coast North America'].shift(-8)
subset_only_corr['PCI- West Coast North America'] = subset_only_corr['PCI- West Coast North America'].shift(+4)
subset_only_corr['PCI- United Kingdom/Continent'] = subset_only_corr['PCI- United Kingdom/Continent'].shift(-4)
subset_only_corr['PCI- South East Asia'] = subset_only_corr['PCI- South East Asia'].shift(+8)


subset_only_corr.dropna(axis=0, inplace=True)
subset_only_corr.reset_index(drop=True,inplace=True)
subset_only_corr.to_csv("subset_only_corr.csv", index = False)

#%%
# ONLY DTW
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_only_dtw = data[['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'Average Earnings',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)']]                   

subset_only_dtw['Average Earnings'] = subset_only_dtw['Average Earnings'].shift(+8)
subset_only_dtw['Newbuilding Prices(13000/14000 TEU)'] = subset_only_dtw['Newbuilding Prices(13000/14000 TEU)'].shift(-4)
subset_only_dtw['Newbuilding Prices(3500/4000 TEU)'] = subset_only_dtw['Newbuilding Prices(3500/4000 TEU)']
subset_only_dtw['Newbuilding Prices(13000/13500 TEU)'] = subset_only_dtw['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_only_dtw['PCI-Comprehensive'] = subset_only_dtw['PCI-Comprehensive'].shift(+8)
subset_only_dtw['PCI- East Coast North America'] = subset_only_dtw['PCI- East Coast North America'].shift(-8)
subset_only_dtw['PCI- West Coast North America'] = subset_only_dtw['PCI- West Coast North America'].shift(+8)
subset_only_dtw['PCI- United Kingdom/Continent'] = subset_only_dtw['PCI- United Kingdom/Continent'].shift(-8)
subset_only_dtw['PCI- Mediterranean/Black Sea'] = subset_only_dtw['PCI- Mediterranean/Black Sea'].shift(-8)


subset_only_dtw.dropna(axis=0, inplace=True)
subset_only_dtw.reset_index(drop=True,inplace=True)
subset_only_dtw.to_csv("subset_only_dtw.csv", index = False)

#%%
# ONLY img2vec
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_only_img2vec = data[['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom/Continent',
       'PCI- Mediterranean/Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']]                   

subset_only_img2vec['Average Earnings'] = subset_only_img2vec['Average Earnings'].shift(-8)
subset_only_img2vec['Bunker Prices'] = subset_only_img2vec['Bunker Prices'].shift(-8)
subset_only_img2vec['Total Container ships Number'] = subset_only_img2vec['Total Container ships Number'].shift(+8)
subset_only_img2vec['Total Container ships TEU'] = subset_only_img2vec['Total Container ships TEU'].shift(+4)
subset_only_img2vec['Newbuilding Prices(1650/1850 TEU)'] = subset_only_img2vec['Newbuilding Prices(1650/1850 TEU)'].shift(+4)
subset_only_img2vec['Newbuilding Prices(13000/14000 TEU)'] = subset_only_img2vec['Newbuilding Prices(13000/14000 TEU)']
subset_only_img2vec['Newbuilding Prices(3500/4000 TEU)'] = subset_only_img2vec['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
subset_only_img2vec['Newbuilding Prices(13000/13500 TEU)'] = subset_only_img2vec['Newbuilding Prices(13000/13500 TEU)'].shift(+8)
subset_only_img2vec['5 Year Finance based on Libor'] = subset_only_img2vec['5 Year Finance based on Libor'].shift(-4)
subset_only_img2vec['PCI-Comprehensive'] = subset_only_img2vec['PCI-Comprehensive'].shift(+8)
subset_only_img2vec['PCI- East Coast North America'] = subset_only_img2vec['PCI- East Coast North America']
subset_only_img2vec['PCI- West Coast North America'] = subset_only_img2vec['PCI- West Coast North America'].shift(-8)
subset_only_img2vec['PCI- United Kingdom/Continent'] = subset_only_img2vec['PCI- United Kingdom/Continent']
subset_only_img2vec['PCI- Mediterranean/Black Sea'] = subset_only_img2vec['PCI- Mediterranean/Black Sea'].shift(+8)
subset_only_img2vec['PCI- East Asia'] = subset_only_img2vec['PCI- East Asia'].shift(+4)
subset_only_img2vec['PCI- South East Asia'] = subset_only_img2vec['PCI- South East Asia'].shift(-8)
subset_only_img2vec['PCI- China P.R.'] = subset_only_img2vec['PCI- China P.R.'].shift(-8)

subset_only_img2vec.dropna(axis=0, inplace=True)
subset_only_img2vec.reset_index(drop=True,inplace=True)
subset_only_img2vec.to_csv("subset_only_img2vec.csv", index = False)
#%%
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

last_subset = data[['SCFI', 'PCI-Comprehensive', 'PCI- West Coast North America', 
                           'Average Earnings',
        'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']]                   

last_subset['Average Earnings'] = last_subset['Average Earnings'].shift(-8)
last_subset['Total Container ships Number'] = last_subset['Total Container ships Number'].shift(+8)
last_subset['Total Container ships TEU'] = last_subset['Total Container ships TEU'].shift(+4)
last_subset['Newbuilding Prices(1650/1850 TEU)'] = last_subset['Newbuilding Prices(1650/1850 TEU)'].shift(+4)
last_subset['Newbuilding Prices(13000/14000 TEU)'] = last_subset['Newbuilding Prices(13000/14000 TEU)'].shift(-8)
last_subset['Newbuilding Prices(3500/4000 TEU)'] = last_subset['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
last_subset['Newbuilding Prices(13000/13500 TEU)'] = last_subset['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
last_subset['5 Year Finance based on Libor'] = last_subset['5 Year Finance based on Libor'].shift(-4)
last_subset['PCI-Comprehensive'] = last_subset['PCI-Comprehensive'].shift(+8)
last_subset['PCI- West Coast North America'] = last_subset['PCI- West Coast North America'].shift(-8)


last_subset.dropna(axis=0, inplace=True)
last_subset.reset_index(drop=True,inplace=True)
last_subset.to_csv("last_subset.csv", index = False)
#%%
# 비교대상 subset 만들기
# 20%  only mse

data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_20_only_mse = data[['SCFI',
        'Average Earnings',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)']]


subset_20_only_mse['Average Earnings'] = subset_20_only_mse['Average Earnings'].shift(-4)
subset_20_only_mse['Newbuilding Prices(13000/13500 TEU)'] = subset_20_only_mse['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_20_only_mse['Newbuilding Prices(3500/4000 TEU)'] = subset_20_only_mse['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
subset_20_only_mse['Newbuilding Prices(13000/14000 TEU)'] = subset_20_only_mse['Newbuilding Prices(13000/14000 TEU)'].shift(-8)

subset_20_only_mse.dropna(axis=0, inplace=True)
subset_20_only_mse.reset_index(drop=True,inplace=True)
subset_20_only_mse.to_csv("subset_20_only_mse.csv", index = False)

#%%
# 비교대상 subset 만들기
# 20%  only dtw

data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_20_only_dtw = data[['SCFI', 'PCI-Comprehensive', 'Average Earnings',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)']]                   

subset_20_only_dtw['Average Earnings'] = subset_20_only_dtw['Average Earnings'].shift(+8)

subset_20_only_dtw['Newbuilding Prices(3500/4000 TEU)'] = subset_20_only_dtw['Newbuilding Prices(3500/4000 TEU)']
subset_20_only_dtw['Newbuilding Prices(13000/13500 TEU)'] = subset_20_only_dtw['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_20_only_dtw['PCI-Comprehensive'] = subset_20_only_dtw['PCI-Comprehensive'].shift(+8)


subset_20_only_dtw.dropna(axis=0, inplace=True)
subset_20_only_dtw.reset_index(drop=True,inplace=True)
subset_20_only_dtw.to_csv("subset_20_only_dtw.csv", index = False)
#%%
# 20% ONLY img2vec
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_20_only_img2vec = data[['SCFI',
       'PCI- West Coast North America', 'Average Earnings',
        'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650/1850 TEU)',
       'Newbuilding Prices(13000/14000 TEU)',
       'Newbuilding Prices(3500/4000 TEU)',
       'Newbuilding Prices(13000/13500 TEU)', '5 Year Finance based on Libor']]                   

subset_20_only_img2vec['Average Earnings'] = subset_20_only_img2vec['Average Earnings'].shift(-8)
subset_20_only_img2vec['Total Container ships Number'] = subset_20_only_img2vec['Total Container ships Number'].shift(+8)
subset_20_only_img2vec['Total Container ships TEU'] = subset_20_only_img2vec['Total Container ships TEU'].shift(+4)
subset_20_only_img2vec['Newbuilding Prices(1650/1850 TEU)'] = subset_20_only_img2vec['Newbuilding Prices(1650/1850 TEU)'].shift(+4)
subset_20_only_img2vec['Newbuilding Prices(13000/14000 TEU)'] = subset_20_only_img2vec['Newbuilding Prices(13000/14000 TEU)']
subset_20_only_img2vec['Newbuilding Prices(3500/4000 TEU)'] = subset_20_only_img2vec['Newbuilding Prices(3500/4000 TEU)'].shift(+8)
subset_20_only_img2vec['Newbuilding Prices(13000/13500 TEU)'] = subset_20_only_img2vec['Newbuilding Prices(13000/13500 TEU)'].shift(+8)
subset_20_only_img2vec['5 Year Finance based on Libor'] = subset_20_only_img2vec['5 Year Finance based on Libor'].shift(-4)
subset_20_only_img2vec['PCI- West Coast North America'] = subset_20_only_img2vec['PCI- West Coast North America'].shift(-8)


subset_20_only_img2vec.dropna(axis=0, inplace=True)
subset_20_only_img2vec.reset_index(drop=True,inplace=True)
subset_20_only_img2vec.to_csv("subset_20_only_img2vec.csv", index = False)

#%%
# ONLY CORR
data = pd.read_csv('C:/Users/hyunj/.spyder-py3/paper/subset_2018_full.csv')
data.dropna(axis=0, inplace=True)

subset_20_only_corr = data[['SCFI',
                        'Newbuilding Prices(13000/13500 TEU)', 'Average Earnings', 'Newbuilding Prices(3500/4000 TEU)', 
                        'Newbuilding Prices(13000/14000 TEU)','PCI- East Coast North America',
                        'PCI- West Coast North America','PCI- United Kingdom/Continent']]                        

subset_20_only_corr['Newbuilding Prices(13000/13500 TEU)'] = subset_20_only_corr['Newbuilding Prices(13000/13500 TEU)'].shift(-8)
subset_20_only_corr['Average Earnings'] = subset_20_only_corr['Average Earnings'].shift(-8)
subset_20_only_corr['Newbuilding Prices(3500/4000 TEU)'] = subset_20_only_corr['Newbuilding Prices(3500/4000 TEU)'].shift(-8)
subset_20_only_corr['Newbuilding Prices(13000/14000 TEU)'] = subset_20_only_corr['Newbuilding Prices(13000/14000 TEU)'].shift(-8)
subset_20_only_corr['PCI- East Coast North America'] = subset_20_only_corr['PCI- East Coast North America'].shift(-8)
subset_20_only_corr['PCI- West Coast North America'] = subset_20_only_corr['PCI- West Coast North America'].shift(+4)
subset_20_only_corr['PCI- United Kingdom/Continent'] = subset_20_only_corr['PCI- United Kingdom/Continent'].shift(-4)


subset_20_only_corr.dropna(axis=0, inplace=True)
subset_20_only_corr.reset_index(drop=True,inplace=True)
subset_20_only_corr.to_csv("subset_20_only_corr.csv", index = False)