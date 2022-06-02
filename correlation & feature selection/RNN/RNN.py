def simple_rnn_model(X_train, y_train, X_test, ㄴㅊ):
    """
    create single layer rnn model trained on X_train and y_train 
    and make predictions on the X_test data 
    """
    # create a model 
    from keras.models import Sequential 
    from keras.layers import Dense, SimpleRNN 
    
    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(256, activation = 'relu'))

    my_rnn_model.add(Dense(4)) # The time step of the output 
    
    my_rnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # fit the RNN model 
    my_rnn_model.fit(X_train, y_train, epochs = 100, batch_size = 8, verbose = 2) 
    
    # predict 
    rnn_predictions = my_rnn_model.predict(X_test)
    
    from sklearn.preprocessing import MinMaxScaler 
    # 학습을 위해 스케일링했던 것을 다시 돌려 놓기
    rnn_predictions = sc.inverse_transform(rnn_predictions)

    return my_rnn_model, rnn_predictions

#%%
# 합치고 분리 7:3 shuffle = flase

def ts_train_test_normalize(all_data, time_steps=4, for_periods=4):
    """
    input: 
        data: dataframe with dates and price data
    output: 
        X_train, y_train: data from 2013/1/1-2018/12/31 
        X_test : data from 2019- 
        sc :     insantiated MinMaxScaler object fit to the training data 
    """
    all_data = data
    X_data = all_data[['PCI-Comprehensive',
       'PCI- East Coast North America', 'PCI- West Coast North America',
       'PCI- United Kingdom/Continent', 'PCI- Mediterranean/Black Sea',
       'PCI- East Asia', 'PCI- South East Asia', 'PCI- China P.R.',
       'Average Earnings', '연료유가',
       'Total Containerships - % Idle/Laid Up/Scrubber Retrofit',
       'Total Containerships - % Idle/Laid Up/Scrubber Retrofit.1',
       '신조선가-1,650/1,850 TEU', '신조선가-13,000/14,000 TEU',
       '신조선가-3,500/4,000 TEU', '신조선가-13,000/13,500', '이자율']].values
    y_data = all_data['SCFI'].values

    # 역스케일링해주기 위해서 scaling(?)을 리턴해주기 위함
    from sklearn.preprocessing import MinMaxScaler 
    sc = MinMaxScaler(feature_range=(0,1))
    sc.fit_transform(all_data['Original SCFI'].values.reshape(-1, 1))
    
    
    # create training and test set
    # create training data of s samples and t time steps 
    X = [] 
    y = [] 
    for i in range(time_steps, len(all_data)-1):
        print(i)
        # 이걸 해준 이유는 i 시점에서 for_periods 만큼의 time step이 더 없는데 해주게 되면 오류가 발생
        if(i + for_periods > len(all_data)) :
            break
        X.append(X_data[i-time_steps : i])
        y.append(y_data[i : i+for_periods])
        
    # 여기까지가 전체 데이터를 합친 것(그때 얘기한)
    X, y = np.array(X), np.array(y)
    #X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    y_test = sc.inverse_transform(y_test)
    return X_train, y_train , X_test, y_test, sc 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/USER/Dropbox/project code/KMI container/data/0525/subset_2018_full.csv')
data = data.iloc[:,1:-1]

X_train, y_train, X_test, y_test, sc = ts_train_test_normalize(data, 4, 4)
my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)

#%%
# predict period가 4기 때문에 뒤의 3개의 값이 누락돼서 이렇게 구현했습니다.

plt.figure(figsize=(12, 9))
plt.plot(y_test, label='actual')
plt.plot(rnn_predictions_2, label='prediction')
plt.legend()
plt.show()
#%%
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape 
    
print('MAPE : ', MAPE(y_test, rnn_predictions_2))
#%%
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, rnn_predictions_2)) 