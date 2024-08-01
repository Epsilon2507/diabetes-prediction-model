# diabetes-prediction-model


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm


diabetes_dataset=pd.read_csv('/content/diabetes.csv')
diabetes_dataset.describe()
diabetes_dataset.groupby('Outcome').mean()

#seprating data set

X= diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
print(X)
scaler=StandardScaler()
standardized_data=scaler.transform(X)
print(standardized_data)
X=standardized_data
Y=diabetes_dataset['Outcome']
print(X)
print(Y )
#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print (X.shape,X_train.shape,X_test.shape)

# training the model

classifier =svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#accuracy score on training  data 

X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print('accuracy score of the training data:',training_data_accuracy)
input_data=(4,110,92,0,0,37.6,0.191,30)
#changing input data intu numpyarray
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standardize inputdata
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)




final result =[[ 0.04601433 -0.34096773  1.18359575 -1.28821221 -0.69289057  0.71168975
  -0.84827977 -0.27575966]]
[0]
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:465: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
