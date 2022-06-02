import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv('2018.csv')

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

# SCFI 변동성을 담기 위한 리스트
SCFI_variability = []

# 데이터프레임의 SCFI column의 i행과 i+1 행의 변동성 확인하기
for i in range(len(data['SCFI'])-1) :
    temp = data.iloc[i, 1]
    temp2 = data.iloc[i+1, 1]
    variability = round(temp2 / temp * 100,3)
    SCFI_variability.append(variability)

sns.distplot(SCFI_variability, kde=True, rug=True)
plt.title("Distrubtion of SCFI variability")
plt.show()

# 변동성을 Long common subsequence에 할 수 있도록 데이터를 문자열로 변환하기
# elif를 사용할 수 있지만 시각적으로 이해를 위해 if를 반복 사용
def Preproessing_variablity(var_list) :
    preprocessed_var = ""
    for var in var_list :
        if 100 <= var < 102.5 :
            preprocessed_var += "A"
            continue
        if 102.5 <= var < 105 :
            preprocessed_var += "B"
            continue
        if 105 <= var < 107.5 :
            preprocessed_var += "C"
            continue
        if 107.5 <= var < 110 :
            preprocessed_var += "D"
            continue
        if 110 <= var < 112.5 :
            preprocessed_var += "E"
            continue
        if 97.5 <= var < 100 :
            preprocessed_var += "a"
            continue
        if 95 <= var < 97.5 :
            preprocessed_var += "b"
            continue
        if 92.5 <= var < 95 :
            preprocessed_var += "c"
            continue
        if 90 <= var < 92.5 :
            preprocessed_var += "d"
            continue
        else :
            preprocessed_var += "F"
            continue
    return preprocessed_var

#  범위를 더 좁게 
def Preproessing_variablity2(var_list) :
    preprocessed_var = ""
    for var in var_list :
        if 100 <= var < 101 :
            preprocessed_var += "A"
            continue
        if 101 <= var < 102 :
            preprocessed_var += "B"
            continue
        if 102 <= var < 103 :
            preprocessed_var += "C"
            continue
        if 103 <= var < 104 :
            preprocessed_var += "D"
            continue
        if 104 <= var < 105 :
            preprocessed_var += "E"
            continue
        if 105 <= var < 106 :
            preprocessed_var += "F"
            continue
        if 106 <= var < 107 :
            preprocessed_var += "G"
            continue
        if 107 <= var < 108 :
            preprocessed_var += "H"
            continue
        if 108 <= var < 109 :
            preprocessed_var += "I"
            continue
        if 109 <= var < 110 :
            preprocessed_var += "J"
            continue
        if 110 <= var < 111 :
            preprocessed_var += "K"
            continue
        if 99 <= var < 100 :
            preprocessed_var += "a"
            continue
        if 98 <= var < 99 :
            preprocessed_var += "b"
            continue
        if 97 <= var < 98 :
            preprocessed_var += "c"
            continue
        if 96 <= var < 97 :
            preprocessed_var += "d"
            continue
        if 95 <= var < 96 :
            preprocessed_var += "e"
            continue
        if 94 <= var < 95 :
            preprocessed_var += "f"
            continue
        if 93 <= var < 94 :
            preprocessed_var += "h"
            continue
        if 92 <= var < 93 :
            preprocessed_var += "i"
            continue
        if 91 <= var < 92 :
            preprocessed_var += "j"
            continue
        if 90 <= var < 91 :
            preprocessed_var += "k"
            continue
        else :
            preprocessed_var += "X"
            continue
    return preprocessed_var

#%%
AE_variability = []

# 데이터프레임의 SCFI column의 i행과 i+1 행의 변동성 확인하기
for i in range(len(data['Average Earnings'])-1) :
    temp = data.iloc[i, 10]
    temp2 = data.iloc[i+1, 10]
    variability = round(temp2 / temp * 100,3)
    AE_variability.append(variability)

sns.distplot(AE_variability, kde=True, rug=True)
plt.title("Distrubtion of Average Earnings variability")
plt.show()

#%%
# Long Common Subsequence 구하기
def lcs_dp (x, y):
	# create a table for dynamic programming of size 
	# (len(x)+1)x(len(y)+1)
	dp_table = np.zeros((len(x)+1, len(y)+1))
	# solve the problem in a bottom up manner
	for i in range(1,len(x)+1):
		for j in range(1, len(y)+1):
			if x[i-1] == y[j-1]:
				dp_table[i-1, j-1] = 1 + dp_table[i-2, j-2]
			else:
				dp_table[i-1, j-1] = max(dp_table[i-1, j-2], 
								dp_table[i-2, j-1])
	return dp_table[len(x)-1, len(y)-1]

ret = lcs_dp(Preproessing_variablity(SCFI_variability), Preproessing_variablity(AE_variability))
print(ret)
#%%
# 다른 것과 비교
temp_variability = []

# 데이터프레임의 SCFI column의 i행과 i+1 행의 변동성 확인하기
for i in range(len(data['Average Earnings'])-1) :
    temp = data.iloc[i, 19]
    temp2 = data.iloc[i+1, 19]
    variability = round(temp2 / temp * 100,3)
    temp_variability.append(variability)
ret = lcs_dp(Preproessing_variablity(SCFI_variability), Preproessing_variablity(temp_variability))
print(ret)
#%%
def variability_list(column):
    var_list = []
    for i in range(len(data[column])-1) : 
        temp = data[column].iloc[i]
        temp2 = data[column].iloc[i+1]
        variability = round(temp2 / temp * 100,3)
        var_list.append(variability)
    
    return var_list

target = Preproessing_variablity2(SCFI_variability)

for column in columns :
    #sns.distplot(variability_list(column), kde=True, rug=True)
    #plt.title("Distrubtion of "+ column +" variability")
    #plt.show()
    feature = Preproessing_variablity2(variability_list(column))
    #print(column, '의 변환된 데이터는 ', feature)
    print(column , "와 SCFI의 LCS는 ", lcs_dp(feature, target))