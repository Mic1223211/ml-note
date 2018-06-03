import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import quandl 
import scipy.stats as st 
quandl.ApiConfig.api_key = 'tAyfv1zpWnyhmDsp91yv'
spy_table = quandl.get('BCTW/_SPXT')
spy_total = spy_table[['Open','Close']]
#calculate log returns
spy_log_return = np.log(spy_total.Close).diff().dropna()

print('Population mean',np.mean(spy_log_return))
print('Population standard deviation:',np.std(spy_log_return))

print('10 days sample returns :', np.mean(spy_log_return.tail(10)))
print('10 days sample standard deviation :', np.std(spy_log_return.tail(10)))
print('1000 days sample returns :',np..mean(spy_log_return.tail(1000)))
print('1000 days sample standard deviation:',np.std(spy_log_return.tail(1000)))

#apply the formula above to calculate confidence interval
bottom_1 = np.mean(spy_log_return.tail(10)) - 1.96*np.std(spy_log_return.tail(10))/(np.sqrt(len((spy_log_return.tail(10)))))
upper_1 = np.mean(spy_log_return.tail(10)) + 1.96*np.std(spy_log_return.tail(10))/(np.sqrt(len((spy_log_return.tail(10)))))

bottom_2 = np.mean(spy_log_return.tail(1000)) - 1.96*np.std(spy_log_return.tail(1000))/(np.sqrt(len((spy_log_return.tail(1000)))))
upper_2 = np.mean(spy_log_return.tail(1000)) + 1.96*np.std(spy_log_return.tail(1000))/(np.sqrt(len((spy_log_return.tail(1000)))))

#print
print('10 days 95% confidence inverval:',(bottom_1,upper_1))
print('1000 days 95% confidence inverval:',(bottom_2,upper_2))

mean_1000 = np.mean(spy_log_return.tail(1000))
std_1000 = np.std(spy_log_return.tail(1000))
mean_10 = np.mean(spy_log_return.tail(10))
std_10 = np.std(spy_log_return.tail(10))
s = pd.Series([mean_10,std_10,mean_1000,std_1000],index=['mean_10','std_10','mean_1000','std_1000'])

print(s)

bottom = 0 -1.64*std_1000/np.sqrt(1000)
upper = 0 + 1.64*std_1000/np.sqrt(1000)
print(bottom,upper)

bottom = 0 -1.96*std_1000/np.sqrt(1000)
upper = 0 + 1.96*std_1000/np.sqrt(1000)
print(np.sqrt(1000)*(mean_1000 -0)/std_1000)


print(1 - st.norm.cdf(1.9488))

mean_1200 = np.mean(spy_log_return.tail(1200))
std_1200 = np.std(spy_log_return.tail(1200))
z_score = np.sqrt(1200)*(mean_1200 -0)/std_1200

print('z-score = ',z_score)
p_value = (1 - st.norm.cdf(z_score))
print('p_value = ',p_value)
