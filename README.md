# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start.

Step 2: Import the necessary libraries and read the file student scores.

Step 3: Print the x and y values.

Step 4: Separate the independent values and dependent values.

Step 5: Split the data.

Step 6: Create a regression model.

Step 7: Find MSE, MAE, RMSE and print the values.

Step 8: Stop.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Cynthia Mehul J

RegisterNumber: 212223240020
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_test,regressor.predict(x_test),color='yellow')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![image](https://github.com/user-attachments/assets/b7a13170-8667-4d2c-b794-fdec5ffb8ab1)
![image](https://github.com/user-attachments/assets/f00ff149-e4f2-4a81-af27-bfcb3793abcc)
![image](https://github.com/user-attachments/assets/ed890c86-cbd2-4b52-b680-c0b0c47275dc)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
