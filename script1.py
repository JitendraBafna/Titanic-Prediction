cols = 12 #Number of columns(features) in train data file


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

#path for train data file
path_train = "/home/bafna/Desktop/ML/KAGGLE/titanic/train.csv";
path_test = "/home/bafna/Desktop/ML/KAGGLE/titanic/test.csv";
path_out = "/home/bafna/Desktop/ML/KAGGLE/titanic/output.csv";
#list of intergers(index) which we want to select or import from training file
drop_col_train = [11,10,8,3,1,0];
drop_col_test = [10,9,7,2,0];

l = []

for i in range(0,cols):
	l.append(i);


data_train = pd.read_csv(path_train, header=None,usecols=l,skiprows=1)
#target = pd.read_csv(path_train, header=None,skiprows=1,usecols=[80])
data_test = pd.read_csv(path_test, header=None,skiprows=1)

m,n = data_train.shape
p,q = data_test.shape

data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)
target = data_train[1].as_matrix()
#target = target.reshape((m,1))
ids = data_test[0].as_matrix()
#ids = ids.reshape((p,1))
print("Ids shape",ids.shape)


print("target shape : ",target.shape)

for x in drop_col_train:
	del data_train[x]

for x in drop_col_test:
	del data_test[x];

data_train[4].replace(['female','male'],[1,0],inplace=True)
#data_train[4].replace('female',1,inplace=True)
data_test[3].replace(['female','male'],[1,0],inplace=True)
#data_test[3].replace('female',1,inplace=True)

data_train.replace(np.nan,0,inplace=True)
data_test.replace(np.nan,0,inplace=True)


#print(data_train[4])
print(data_train[:1])
print(data_test[:1])
print("Train file shape is : ",data_train.shape)
print("Test file shape is : ",data_test.shape)

data_train = data_train.as_matrix()
data_test = data_test.as_matrix()

print("Train file shape is : ",data_train.shape)
print("Test file shape is : ",data_test.shape)


model = LogisticRegression()
model.fit(data_train,target,sample_weight=None)
predicted = model.predict(data_test)

p = predicted.shape
print(p)
#predicted = np.asmatrix(predicted)
#predicted = np.reshape(predicted,(p,1))

output = np.column_stack((ids,predicted))
#out = out.reshape((p,2))
print(output.shape)
np.savetxt(path_out,output,delimiter=",",header="PassengerId,Survived", comments='', fmt='%d,%d')