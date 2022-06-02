# 추세에 집중 !
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv('2018.csv')
print(data)
#%%
# 'Date' Column이 object임을 확인
data.info()
#%%
# 'Date' Column을 Datetime으로 변환 후 null 값이 없음을 확인
data['Date'] = pd.to_datetime(data['Date'])
data.info()
#%%
# LAG 추가 ★상재형이 첨부한 그래프보니 target에 대해서는 LAG를 적용하지 않았음
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
# 105개가 맞음 why? column이 22개가 아니라 datetime column을 빼서 21 => 21 * 5 = 105
print('new_columns의 개수는 '+str(len(new_columns)))
#%%
# 그래프 jpg 파일로 저장
for column in new_columns:
    # '/' 이게 있으면 코드에서 오류가 나서 csv파일에서 '/'을 삭제함
    data[column].plot()
    # 축 삭제
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)    
    ax.axes.yaxis.set_visible(False)
    file_name = '.\graph\\'+ column + '.jpg'
    plt.savefig(file_name)
    plt.show()
    
#%%

# 이미지 유사도 비교
from skimage.metrics import structural_similarity as ssim 
import imutils
import cv2
from skimage import io

# 예시 : 두 그래프가 유사하지 않음에도 SSIM 지수가 높게 나옴
imageA = cv2.imread("graph_2018/SCFI.jpg")
imageB = cv2.imread("graph/Total Container ships Number with +4 week lag.jpg")

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

cv2.imshow('diff',diff)
cv2.waitKey(0)
# Total Container ships Number with 4 week lag와 Original SCFI와 유사도는 0.8867692192138871이다.
# ★ SSIM 지수는 활용하지 못함을 의미 
print("SSIM: {}".format(score))
#%%

# SSIM을 활용하지 못하니 MSE를 사용
# MSE 함수 정의
def mse(imageA, imageB):
    ret, thres1 = cv2.threshold(imageA, 150, 255, cv2.THRESH_BINARY_INV)
    ret, thres2 = cv2.threshold(imageB, 150, 255, cv2.THRESH_BINARY_INV)
    err = np.sum((thres1.astype("float") - thres2.astype("float")) ** 2)
    err /= float(thres1.shape[0] * thres1.shape[1])
    return err

#%%

# MSE를 활용하기 전에 각 그래프의 이미지의 픽셀 구성을 확인
image_pixel = cv2.imread("graph/SCFI.jpg")
print(image_pixel.shape) #(288, 432, 3)
print(np.unique(image_pixel))

# ★ 채널이 3개기 때문에 이를 흑백으로 변환해줘야 더욱 정확한 MSE 값을 측정할 것임
# 흑백변환 후 이미지 픽셀 확인
gray_image = cv2.cvtColor(image_pixel, cv2.COLOR_BGR2GRAY)
print(gray_image.shape) #(288, 423)
print(np.unique(gray_image))
# OpenCV를 활용한 정규화
# 정규화 함수를 사용하면 집중되어 있던 히스토그램이 균일하게 분포 -> 우리는 필요가 없음
#img_norm2 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
#print(img_norm2.shape)
#print(np.unique(img_norm2))

#cv2.imshow('diff',img_norm2)
#cv2.waitKey(0)
#%%

# 내가 생각하기에 필요한 것은 그 픽셀에 선이 있으면 검은색, 없으면 하얀색으로해서 정확하게 비교하는 것이 필요하다고 생각
# Thresholding을 활용 : Threshold을 기준으로 검은색(0), 하얀색(255)로 결정
# INV를 활용해서 검정과 하얀색을 바꿔준다(검은 바탕에 흰선) 
# 최대 값을 255로 하면 MSE값이 너무 커질테니 값보다 크면 하얀색을 1로 설정 
image_pixel = cv2.imread("graph/SCFI.jpg")
gray_image = cv2.cvtColor(image_pixel, cv2.COLOR_BGR2GRAY)
cv2.imshow('black', gray_image)
ret, thres = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
print(thres.shape)
print(ret)
print(np.unique(thres))
cv2.imshow('threshold', thres)
cv2.waitKey(0)

#%%
# 다른 경우의 MSE는 442.72009227109055
imageA = cv2.imread("graph_2018/SCFI.jpg")
imageB = cv2.imread("graph/Total Container ships Number with +4 week lag.jpg")

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

m = mse(grayA, grayB)
print(m)
#%%
# 서로 비슷한 경우의 MSE는 164.8968058770576
imageA = cv2.imread("graph_2018/Average Earnings.jpg")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
ret, thres3 = cv2.threshold(grayA, 150, 255, cv2.THRESH_BINARY_INV)
print(len(thres3[thres3==255]))




#%%
# 서로 완전히 똑같은 경우의 MSE는 0
imageA = cv2.imread("graph_2018/SCFI.jpg")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
ret, thres2 = cv2.threshold(grayA, 150, 255, cv2.THRESH_BINARY_INV)
print(len(thres2[thres2==255]))
#cv2.imshow('threshold', thres2)
#cv2.waitKey(0)

#%%
# 이미지의 반전을 mse가 허용하는지 판단하기
# 이미지의 반전을 허용하지 않음 

print(mse(image_target_reversed, grayA))
#%%
# SSIM은 구조적 유사성으로 다르더라도 알고리즘에 따라 수치가 높게 나올 수 있음.
# 그래서 MSE를 사용해서 이미지 유사도 측정

# 그 전에 TARGET 값 지정 
target = cv2.imread('graph_2018/SCFI.jpg', cv2.IMREAD_COLOR)
target_reversed = cv2.flip(target, 0) # 상하반전
image_target_reversed = cv2.cvtColor(target_reversed, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(image_target_reversed, 150, 255, cv2.THRESH_BINARY_INV)
print(len(thres[thres==255]))
cv2.imshow('threshold', thres)
cv2.waitKey(0)
cv2.imwrite("graph/SCFI_Reversed.jpg", target_reversed)

targets = ['SCFI_Reversed', 'SCFI']
new_columns.remove('SCFI')
new_columns.remove('SCFI with -4 week lag')
new_columns.remove('SCFI with +4 week lag')
new_columns.remove('SCFI with -8 week lag')
new_columns.remove('SCFI with +8 week lag')
#%%

# 픽셀이 정말 뒤집어졌는지 확인
Image_SCFI_Original = cv2.imread('graph/SCFI.jpg', cv2.IMREAD_COLOR)
Image_SCFI_Original = cv2.cvtColor(Image_SCFI_Original, cv2.COLOR_BGR2GRAY)

#img180 = cv2.rotate(Image_SCFI_Original, cv2.ROTATE_180)
#cv2.imwrite("graph/SCFI_Reversed2.jpg", img180)

Image_SCFI_Reversed = cv2.imread('graph/SCFI_Reversed.jpg', cv2.IMREAD_COLOR)
Image_SCFI_Reversed = cv2.cvtColor(Image_SCFI_Reversed, cv2.COLOR_BGR2GRAY)
                    
#%%
print("확인 : "+str(len(new_columns)))
#%%

# 모든 Feature들에 대해서 Target에 대한 MSE 지수 확인 및 적은 feature들 검출
# 
MSE_dic_original = {}
MSE_dic_flipped = {}

for column in tqdm(new_columns) :
    feature_image_name = column+'.jpg'
    feature_path = "graph/" + feature_image_name
    image_feature = cv2.imread(feature_path)
    image_feature_gray = cv2.cvtColor(image_feature, cv2.COLOR_BGR2GRAY)
    for target in targets :
        target_image_name = target+'.jpg'
        target_path = "graph/" + target_image_name
        image_target = cv2.imread(target_path)
        image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
        
        MSE = round(mse(image_feature_gray, image_target_gray),2)
        if(target == 'SCFI') :
            MSE_dic_original[column] = MSE;
        else:
            MSE_dic_flipped[column] = MSE;
            
#%%

# 각 딕셔너리를 MSE에 대해서 오름차순으로 정리하고 작은 것부터 10개를 가져온다.
sorted_MSE_dic_original = sorted(MSE_dic_original.items(), key=lambda x : x[1])
sorted_MSE_dic_flipped = sorted(MSE_dic_flipped.items(), key=lambda x : x[1])

# 먼저 sorted_MSE_dic_original에서 작은 값 10개를 가져오겠다.
# 작은 것들 확인해보면 그래프가 어느정도 유사한 것을 확인할 수는 있음
print("Original 버전")
for i in range(len(sorted_MSE_dic_original)) :
    print(sorted_MSE_dic_original[i][0] + "의 MSE 값은 " + str(sorted_MSE_dic_original[i][1])+ "이다.\n")

# ★★★★★Flippend의 MSE값은 형편 없음 => Flipped에서 뭔가 이상함 내 눈엔 달라보이는데 mse가 작음
print("#Flipped 버전")
for i in range(len(sorted_MSE_dic_flipped)) :
    print(sorted_MSE_dic_flipped[i][0] + "의 MSE 값은 " + str(sorted_MSE_dic_flipped[i][1])+ "이다.\n")
    #%%
#data['Date with +4 week lag'] = data.query('"2018-02-02"<= Date')
#data['Date with +8 week lag'] = data.query('"2018-03-02"<= Date')      
#data['Date with -4 week lag'] = data.query('Date<="2021-12-03"')
#data['Date with -8 week lag'] = data.query('Date<="2021-11-05"')   
#%%

# Correlation 비교 (0.8이상 검출)
# Corrleation을 찾을 때 lag를 적용한 것은 그것에 맞게 Target을 달리해준다.(Target을 달리한다는게 시점에 맞게 짜른다!)
data['SCFI with +4 week lag'] = data.loc[4:, ['SCFI']]
data['SCFI with +8 week lag'] = data.loc[8:, ['SCFI']]   
data['SCFI with -4 week lag'] = data.loc[:204, ['SCFI']]
data['SCFI with -8 week lag'] = data.loc[:200, ['SCFI']]

#data['SCFI with +4 week lag'] = data['SCFI'].iloc[[4:], ['SCFI']]

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

# 데이터를 잘 분리했으니 이제 각 lag마다 Correlation을 비교해보자
# Original data 상관관계 분석
corrmat_original = data_original.corr()
sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(corrmat_original, annot=True)

# 전체 SCFI에 대한 상관 관계를 내림차순으로 시각화

corr_target = corrmat_original['SCFI'].reset_index()
corr_target.columns = ['feature','corr']
corr_target = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.75]
ind = np.arange(corr_target.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, corr_target['corr'].values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')
ax.set_xlabel("corr", fontsize = 14)
ax.set_title("Correlations between features and target ", fontsize = 18)
plt.show()

important_value = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.8]
print('SCFI와 상관 관계가 높았던 Feature들은 다음과 같습니다.(Threshold = 0.8)')
for i in range(len(important_value) - 1) :
    print(important_value.iloc[i, 0], important_value.iloc[i, 1].round(2))
#%%
# After 4 lag (4주 후 Shift lag) 상관관계 분석
corrmat_after_4_lag = data_after_4_lag.corr()
sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(corrmat_after_4_lag, annot=True)

# 전체 SCFI에 대한 상관 관계를 내림차순으로 시각화

corr_target = corrmat_after_4_lag['SCFI with +4 week lag'].reset_index()
corr_target.columns = ['feature','corr']
corr_target = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.75]
ind = np.arange(corr_target.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, corr_target['corr'].values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')
ax.set_xlabel("corr", fontsize = 14)
ax.set_title("Correlations between features and target ", fontsize = 18)
plt.show()

important_value = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.8]
print('SCFI +4 week lag 와 상관 관계가 높았던 Feature들은 다음과 같습니다.(Threshold = 0.8)')
for i in range(len(important_value) - 1) :
    print(important_value.iloc[i, 0], important_value.iloc[i, 1].round(2))
#%%
# After 8 lag (8주 후 Shift lag) 상관관계 분석
corrmat_after_8_lag = data_after_8_lag.corr()
sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(corrmat_after_8_lag, annot=True)

# 전체 SCFI에 대한 상관 관계를 내림차순으로 시각화

corr_target = corrmat_after_8_lag['SCFI with +8 week lag'].reset_index()
corr_target.columns = ['feature','corr']
corr_target = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.75]
ind = np.arange(corr_target.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, corr_target['corr'].values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')
ax.set_xlabel("corr", fontsize = 14)
ax.set_title("Correlations between features and target ", fontsize = 18)
plt.show()

important_value = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.8]
print('SCFI +8 week lag 와 상관 관계가 높았던 Feature들은 다음과 같습니다.(Threshold = 0.8)')
for i in range(len(important_value) - 1) :
    print(important_value.iloc[i, 0], important_value.iloc[i, 1].round(2))
#%%
# Before 4 lag (4주 전 Shift lag) 상관관계 분석
corrmat_before_4_lag = data_before_4_lag.corr()
sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(corrmat_before_4_lag, annot=True)

# 전체 SCFI에 대한 상관 관계를 내림차순으로 시각화

corr_target = corrmat_before_4_lag['SCFI with -4 week lag'].reset_index()
corr_target.columns = ['feature','corr']
corr_target = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.75]
ind = np.arange(corr_target.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, corr_target['corr'].values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')
ax.set_xlabel("corr", fontsize = 14)
ax.set_title("Correlations between features and target ", fontsize = 18)
plt.show()

important_value = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.8]
print('SCFI -4 week lag 와 상관 관계가 높았던 Feature들은 다음과 같습니다.(Threshold = 0.8)')
for i in range(len(important_value) - 1) :
    print(important_value.iloc[i, 0], important_value.iloc[i, 1].round(2))
#%%
# Before 8 lag (8주 전 Shift lag) 상관관계 분석
corrmat_before_8_lag = data_before_8_lag.corr()
sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(corrmat_before_8_lag, annot=True)

# 전체 SCFI에 대한 상관 관계를 내림차순으로 시각화

corr_target = corrmat_before_8_lag['SCFI with -8 week lag'].reset_index()
corr_target.columns = ['feature','corr']
corr_target = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.75]
ind = np.arange(corr_target.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,18))
rects = ax.barh(ind, corr_target['corr'].values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')
ax.set_xlabel("corr", fontsize = 14)
ax.set_title("Correlations between features and target ", fontsize = 18)
plt.show()

important_value = corr_target.sort_values(by = 'corr', ascending = True)[:].loc[corr_target['corr'] >0.8]
print('SCFI -8 week lag 와 상관 관계가 높았던 Feature들은 다음과 같습니다.(Threshold = 0.8)')
for i in range(len(important_value) - 1) :
    print(important_value.iloc[i, 0], important_value.iloc[i, 1].round(2))


#%%

# 이제 DTW