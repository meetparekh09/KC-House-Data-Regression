import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import normalize


data = pd.read_csv('kc_house_data.csv')
dates = np.array(data['date'])
dates_int = []

for d in dates:
	dstr = d.split('T')[0]
	dobj = datetime.strptime(dstr, '%Y%m%d')
	dint = int(time.mktime(dobj.timetuple()))
	dates_int.append(dint)


data['date'] = dates_int

#print(data.corr())

data = data.drop(['id', 'date', 'lat', 'long', 'zipcode', 'condition', 'yr_built'], axis=1)
target = np.array(data['price'])
data = np.array(data.drop(['price'], axis=1))

tmean = np.mean(target)
tstd = np.std(target)

print(tmean)
print(tstd)

data = normalize(data)
target = normalize(target).T

np.savetxt('data.txt', data)
np.savetxt('label.txt', target)


