# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Navinkumar v
RegisterNumber:  212223230141
*/
import pandas as pd 
data=pd.read_csv("Placement_Data.csv") 
data.head() 
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()      
data1.duplicated().sum() 
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1] 
x 
y=data1["status"] 
y 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)    #Predicts placement (0 or 1) for the test dataset (x_test).
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print(classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## Data :
![image](https://github.com/user-attachments/assets/536d1234-d7d4-4847-8612-15f43fae64d5)
## Transformed Data:
![image](https://github.com/user-attachments/assets/46d3537a-4400-44c0-b501-395bce804145)
## X and Y values:
![image](https://github.com/user-attachments/assets/c5bee439-0e6f-4a7d-a261-d3a99cba0d42)
![image](https://github.com/user-attachments/assets/bd1fec18-b323-4295-8318-35e88f5257e9)
## Model:
![image](https://github.com/user-attachments/assets/36c5fe9d-7416-43fd-bf36-533de03e4686)
## Accuracy :
![image](https://github.com/user-attachments/assets/290cdae7-1a34-47b3-b0d4-929db48c0c30)
## Classifiaction:
![image](https://github.com/user-attachments/assets/f9ee93c0-8bc3-4392-8242-90a1ef5e83dc)
## Prediction:
![image](https://github.com/user-attachments/assets/fd2ca087-2eb8-4b27-bd77-6d9a234f7f91)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
