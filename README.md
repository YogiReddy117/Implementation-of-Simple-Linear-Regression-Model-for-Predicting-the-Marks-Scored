# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Palleri Yogi
RegisterNumber: 212220040108
*/

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.head()
dataset.tail()

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

#splitting train and test data set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

#graph plot for traing data
plt.scatter(X_train,Y_train,color = "green")
plt.plot(X_train,reg.predict(X_train),color = "red")
plt.title('Training set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color = "blue")
plt.plot(X_test,reg.predict(X_test),color = "silver")
plt.title('Test set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse = mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
1. df.head()
   ![dataset_head](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/592c5505-5ae3-4085-bccd-53284a4b6e7f)

3. df.tail()
   ![dataset_tail](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/bb7d70eb-8200-4965-8e5f-ba2ba41847f7)

6. Array value of X
   ![Array values of X](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/8270530d-0da7-4d6e-b2ba-b2adb240c742)

8. Array value of Y
   ![Array values of Y](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/2e706aeb-c60a-480c-8e47-525d5cf38c08)

10. Values of Y prediction
    ![Y_predicted_values](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/5a8ddf35-9be9-4b1f-b387-d696fc8b28ae)

12. Values of Y test
    ![Y_test_values](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/0e142d44-ea0f-4e5c-acf9-85823eb0a8aa)

14. Training Set Graph
    ![training_set_graph](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/f4aa87d2-5eca-4feb-8736-696a921309fb)

15. Test Set Graph
    ![test_set_graph](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/e44d0633-c017-437e-9446-f399b8199bc0)

17. Values of MSE, MAE and RMSE
    ![error_values](https://github.com/YogiReddy117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123739437/ec8a9a2d-47a6-4ded-a5f2-e96495fafe56)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
