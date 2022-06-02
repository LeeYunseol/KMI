#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 상재형이 연결한 SCFI 기준으로 모두 붙여 놓은 데이터 사용 
data_df =  pd.read_csv('full_data.csv')
print(data_df.shape)
data_df.head()


# In[3]:


# 제 컴퓨터 문제인지는 모르겠지만 heatmap을 그리는데 seaborn / matplotlib 의 시각화 중에서 한글이 깨지는 문제가 발생했습니다.
# 인터넷에 있는 방법을 다 적용해봐도 문제가 있어서 상재 형께서 수정하신 변수명을 다시 영어로 수정하였습니다.

# 변수명 변경
data_df.rename(columns={'용선료':'Clarksons Average Containership Earnings'}, inplace = True)
data_df.rename(columns={'연료유가':'HSFO 380cst Bunker Prices (3.5% Sulphur), Rotterdam'}, inplace = True)
data_df.rename(columns={'신조선가-1,650/1,850 TEU' : 'Containership 1,650/1,850 TEU FCC, G\'less Newbuilding Prices'}, inplace = True)
data_df.rename(columns={'신조선가-13,000/14,000 TEU' : 'Containership 13,000/14,000 TEU Newbuilding Prices'}, inplace = True)
data_df.rename(columns={'신조선가-3,500/4,000 TEU' : 'Containership 3,500/4,000 TEU (Wide Beam) G\'less Newbuilding Prices'}, inplace = True)
data_df.rename(columns={'신조선가-13,000/13,500' : 'Containership 13,000/13,500 TEU G\'less Newbuilding Prices'}, inplace = True)
data_df.rename(columns={'이자율':'5 Year $10m Finance based on Libor 1st yr'}, inplace = True)


# In[4]:


data_df.info()

# Date 컬럼이 Object 타입
# target 컬럼이 전체 SCFI 
# 나머지는 다 float 타입이라 수정할 필요가 없음


# In[5]:


# Date 컬럼을 Timestamp로 변환
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.head()


# In[6]:


# Date 컬럼이 잘 변환됐는지 확인
data_df.info()
# 잘 변환되었음.


# In[7]:


# 2018년 1월 1일 이후 csv파일 만들기
after_2018_data = data_df.query('"2018-01-01"<= Date')
after_2018_data.reset_index(drop=True, inplace=True)
print(after_2018_data.shape)
after_2018_data.head()
# 총 209개 


# In[8]:


after_2018_data.info()


# In[9]:


# 2018년 1월 1일 이후 데이터에 결측치 확인하기
print(after_2018_data.shape)
after_2018_data.isnull().sum()
# 결측치 없음
# 데이터는 총 209개


# In[10]:


# 2018년 이후 데이터에 빠진 주 찾기
from datetime import datetime, timedelta

for i in range(len(after_2018_data)-1):
    if (after_2018_data.loc[i+1,'Date']-after_2018_data.loc[i,'Date'])!=timedelta(days=7):
        diff = after_2018_data.loc[i+1,'Date']-after_2018_data.loc[i,'Date']
        missing_days = diff.days/7
        print(after_2018_data.loc[i,'Date'].strftime('%Y-%m-%d')+'뒤에'+str(missing_days)+'주 만큼 결측')
    
# 결측치 없음(확인해보니 상재형꼐서 다 채우셨음)


# In[11]:


# 2018년 이후 데이터 csv 파일로 저장
after_2018_data.to_csv("data_after_2018.csv", index=False, encoding = 'utf-8-sig')


# In[12]:


# 2020년 1월 1일 이후 csv파일 만들기
after_2020_data = data_df.query('"2020-01-01"<= Date')
after_2020_data.reset_index(drop=True, inplace=True)
print(after_2020_data.shape)
after_2020_data.head()

# 총 105개 


# In[13]:


# 2020년 1월 1일 이후 csv 파일 저장
after_2020_data.to_csv("data_after_2020.csv", index = False, encoding = 'utf-8-sig')


# In[14]:


# 2018년 1월 1일 이후 데이터 상관관계

sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(after_2018_data.corr(), annot=True)


# In[16]:


# 2020년 1월 1일 이후 데이터 상관 관계

sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(after_2020_data.corr(), annot=True)


# In[ ]:




