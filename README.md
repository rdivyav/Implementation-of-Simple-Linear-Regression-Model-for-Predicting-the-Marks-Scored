# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## 1. Prepare the Data
## 2. Split the Data
## 3. Train the Model
## 4. Evaluate and Predict

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divya R V
RegisterNumber:212223100005  
```







```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
```
![Screenshot 2025-02-27 173212](https://github.com/user-attachments/assets/a5496df2-c796-44bb-9ea1-ea53ff779100)




```
df.tail()
```

![Screenshot 2025-02-27 173221](https://github.com/user-attachments/assets/1d6e1563-fa49-4c33-b77e-5e912967c3ee)






```
X=df.iloc[:,:-1].values
X
```

![Screenshot 2025-02-27 173232](https://github.com/user-attachments/assets/5d2c4f5e-024e-4ad6-8b4e-948ae5fe437f)






```
Y=df.iloc[:,1].values
Y
```
![Screenshot 2025-02-27 173244](https://github.com/user-attachments/assets/2fdc4dec-b0ab-4868-8b5e-073ceb725e97)






```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
```






```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```





```
Y_pred
```
![Screenshot 2025-02-27 173252](https://github.com/user-attachments/assets/7e5b9a62-4158-401a-8e64-5199284a425d)






```
Y_test
```
![Screenshot 2025-02-27 173301](https://github.com/user-attachments/assets/98035154-60d2-4e8b-b098-c7de122ea33d)






```
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![Screenshot 2025-02-27 173312](https://github.com/user-attachments/assets/1f70aac9-cea0-40b4-988a-b2d1a828e000)







```
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


![Screenshot 2025-02-27 173323](https://github.com/user-attachments/assets/f20ddc06-48f7-4ff6-890b-8258bc5bdc87)








```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```
![Screenshot 2025-02-27 173333](https://github.com/user-attachments/assets/26f30f70-f648-40d4-a5f6-389fb48acf2a)








## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
