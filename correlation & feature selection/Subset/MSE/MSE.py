import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv('2020.csv')
print(data.columns)
columns = ['SCFI', 'SCFI-Europe (base port)', 'SCFI-Med (base port)',
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

new_columns = ['SCFI', 'SCFI-Europe (base port)', 'SCFI-Med (base port)',
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

lag_list = [-8, -4, +4, +8]
for column in tqdm(columns) :
    for lag in lag_list :
        if (lag > 0) :
            new_column = column + ' with ' + "+" + str(lag) +' week lag'
        else :
            new_column = column + ' with ' + str(lag) +' week lag'
        new_columns.append(new_column)
        data[new_column] = data[column].shift(lag)
new_columns.remove('SCFI')
new_columns.remove('SCFI with -4 week lag')
new_columns.remove('SCFI with +4 week lag')
new_columns.remove('SCFI with -8 week lag')
new_columns.remove('SCFI with +8 week lag')
#%%
# 그래프 jpg 파일로 저장
for column in new_columns:
    # '/' 이게 있으면 코드에서 오류가 나서 csv파일에서 '/'을 삭제함
    data[column].plot()
    # 축 삭제
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)    
    ax.axes.yaxis.set_visible(False)
    file_name = '.\graph_2020\\'+ column + '.jpg'
    plt.savefig(file_name)
    plt.show()
#%%
import cv2
def mse(imageA, imageB):
    ret, thres1 = cv2.threshold(imageA, 150, 255, cv2.THRESH_BINARY_INV)
    ret, thres2 = cv2.threshold(imageB, 150, 255, cv2.THRESH_BINARY_INV)
    err = np.sum((thres1.astype("float") - thres2.astype("float")) ** 2)
    err /= float(thres1.shape[0] * thres1.shape[1])
    return err
#%%   
MSE_dic_original = {}

target = 'SCFI' 
target_image_name = target+'.jpg'
target_path = "graph_2020/" + target_image_name
image_target = cv2.imread(target_path)
image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)

for column in tqdm(new_columns) :
    feature_image_name = column+'.jpg'
    feature_path = "graph_2020/" + feature_image_name
    image_feature = cv2.imread(feature_path)
    image_feature_gray = cv2.cvtColor(image_feature, cv2.COLOR_BGR2GRAY)

    MSE = round(mse(image_feature_gray, image_target_gray),2)

    MSE_dic_original[column] = MSE;

sorted_MSE_dic_original = sorted(MSE_dic_original.items(), key=lambda x : x[1])   

print("Original 버전")
for i in range(len(sorted_MSE_dic_original)) :
    print(sorted_MSE_dic_original[i][0] + "의 MSE 값은 " + str(sorted_MSE_dic_original[i][1])+ "이다.\n")   
#%%
# 박스 플롯 그리기
box_plot = []
for i in range(len(sorted_MSE_dic_original)) :
    box_plot.append(sorted_MSE_dic_original[i][1])
green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(box_plot, flierprops=green_diamond)
plt.title("MSE 2020 box plot")
plt.show()
#%%
# 박스 플롯 값 구하기
data_box_plot = pd.Series(box_plot)
min = np.percentile(data_box_plot,0)      # 최소값
q1 = np.percentile(data_box_plot,25)      # 1사분위수
q2 = np.percentile(data_box_plot,50)      # 2사분위수
q3 = np.percentile(data_box_plot,75)      # 3사분위수
max = np.percentile(data_box_plot,100)  # 최대값
threshold = np.percentile(data_box_plot, 20)

#%%
# 2018년 데이터 분석
print("\n2018년 데이터 분석")
data_2018 = pd.read_csv('2018.csv')
print(data_2018.columns)

column_2018 = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
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

new_columns_2018 = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
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

lag_list = [-8, -4, +4, +8]
for column in tqdm(column_2018) :
    for lag in lag_list :
        if (lag > 0) :
            new_column_2018 = column + ' with ' + "+" + str(lag) +' week lag'
        else :
            new_column_2018 = column + ' with ' + str(lag) +' week lag'
        new_columns_2018.append(new_column_2018)
        data_2018[new_column_2018] = data_2018[column].shift(lag)
        
new_columns_2018.remove('SCFI')
new_columns_2018.remove('SCFI with -4 week lag')
new_columns_2018.remove('SCFI with +4 week lag')
new_columns_2018.remove('SCFI with -8 week lag')
new_columns_2018.remove('SCFI with +8 week lag')
#%%
# 그래프 jpg 파일로 저장
for column in new_columns_2018:
    # '/' 이게 있으면 코드에서 오류가 나서 csv파일에서 '/'을 삭제함
    data_2018[column].plot()
    # 축 삭제
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)    
    ax.axes.yaxis.set_visible(False)
    file_name = '.\graph_2018\\'+ column + '.jpg'
    plt.savefig(file_name)
    plt.show()
#%%   
MSE_dic_original = {}

target = 'SCFI' 
target_image_name = target+'.jpg'
target_path = "graph_2018/" + target_image_name
image_target = cv2.imread(target_path)
image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)

for column in tqdm(new_columns_2018) :
    feature_image_name = column+'.jpg'
    feature_path = "graph_2018/" + feature_image_name
    image_feature = cv2.imread(feature_path)
    image_feature_gray = cv2.cvtColor(image_feature, cv2.COLOR_BGR2GRAY)

    MSE = round(mse(image_feature_gray, image_target_gray),2)

    MSE_dic_original[column] = MSE;

sorted_MSE_dic_original = sorted(MSE_dic_original.items(), key=lambda x : x[1])   

print("Original 버전")
for i in range(len(sorted_MSE_dic_original)) :
    print(sorted_MSE_dic_original[i][0] + "의 MSE 값은 " + str(sorted_MSE_dic_original[i][1])+ "이다.\n")   
#%%
# 박스 플롯 그리기
box_plot = []
for i in range(len(sorted_MSE_dic_original)) :
    box_plot.append(sorted_MSE_dic_original[i][1])
green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(box_plot, flierprops=green_diamond)
plt.title("MSE 2018 box plot")
plt.show()
#%%
# 박스 플롯 값 구하기
data_box_plot = pd.Series(box_plot)
min = np.percentile(data_box_plot,0)      # 최소값
q1 = np.percentile(data_box_plot,25)      # 1사분위수
q2 = np.percentile(data_box_plot,50)      # 2사분위수
q3 = np.percentile(data_box_plot,75)      # 3사분위수
max = np.percentile(data_box_plot,100)  # 최대값
threshold = np.percentile(data_box_plot, 20)

#%% 
# 2018년 Subset csv 파일 만들기
data = pd.read_csv('2018.csv')
subset_2018_mse = data[['Date', 'SCFI', 'Newbuilding Prices(13000,13500 TEU)', 
                        'Average Earnings', 'Newbuilding Prices(3500,4000 TEU)','Newbuilding Prices(13000,14000 TEU)', 'Newbuilding Prices(1650,1850 TEU)']]
# LAG 적용
subset_2018_mse['Newbuilding Prices(13000,13500 TEU)'] = subset_2018_mse['Newbuilding Prices(13000,13500 TEU)'].shift(-8)
subset_2018_mse['Average Earnings'] = subset_2018_mse['Average Earnings'].shift(-4)
subset_2018_mse['Newbuilding Prices(13000,14000 TEU)'] = subset_2018_mse['Newbuilding Prices(13000,14000 TEU)'].shift(-8)
subset_2018_mse['Newbuilding Prices(1650,1850 TEU)'] = subset_2018_mse['Newbuilding Prices(1650,1850 TEU)'].shift(8)
subset_2018_mse.dropna(axis=0, inplace=True)
subset_2018_mse.reset_index(drop=True,inplace=True)
subset_2018_mse.to_csv("subset_2018_mse.csv", index = False)
#%% 
# 2020년 Subset csv 파일 만들기
data = pd.read_csv('2020.csv')
subset_2020_mse = data[['Date', 'SCFI', 'Clarksons Average Containership Earnings', 
                       'Containership 13,000,13,500 TEU G\'less Newbuilding Prices', 'SCFI-Europe (base port)',
                       'SCFI-Med (base port)', 'Containership 13,000,14,000 TEU Newbuilding Prices',
                        'SCFI-EC America (base port)', 'SCFI-S Africa (Durban)','SCFI-ANZ (Melbourne)',
                         'Containership 3,500,4,000 TEU (Wide Beam) G\'less Newbuilding Prices']]
# LAG 적용
subset_2020_mse['Clarksons Average Containership Earnings'] = subset_2020_mse['Clarksons Average Containership Earnings'].shift(-8)
subset_2020_mse['Containership 13,000,13,500 TEU G\'less Newbuilding Prices'] = subset_2020_mse['Containership 13,000,13,500 TEU G\'less Newbuilding Prices'].shift(-4)
subset_2020_mse['Containership 13,000,14,000 TEU Newbuilding Prices'] = subset_2020_mse['Containership 13,000,14,000 TEU Newbuilding Prices'].shift(-8)
subset_2020_mse['SCFI-EC America (base port)'] = subset_2020_mse['SCFI-EC America (base port)'].shift(-8)
subset_2020_mse['SCFI-S Africa (Durban)'] = subset_2020_mse['SCFI-S Africa (Durban)'].shift(-8)
subset_2020_mse['SCFI-ANZ (Melbourne)'] = subset_2020_mse['SCFI-ANZ (Melbourne)'].shift(-4)
subset_2020_mse['Containership 3,500,4,000 TEU (Wide Beam) G\'less Newbuilding Prices'] = subset_2020_mse['Containership 3,500,4,000 TEU (Wide Beam) G\'less Newbuilding Prices'].shift(4)
subset_2020_mse.dropna(axis=0, inplace=True)
subset_2020_mse.reset_index(drop=True,inplace=True)
subset_2020_mse.to_csv("subset_2020_mse.csv", index = False)
