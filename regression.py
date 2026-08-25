import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data={
    "study_hours":[8,10,9,7,6,5,10,11,12,14],
    "Attendance":[70,45,80,67,80,90,85,84,78,83],
    "Result":["pass","Fail","pass","Fail","Fail","pass","pass","pass","pass","Fail"]}
for i in range(len(data["study_hours"])):
 print(f"student{i+1}:study hours:{data['study_hours'][i]},Attendance:{data['Attendance'][i]},Result:{data['Result'][i]}")  
df=pd.DataFrame(data)
print(df)

x=df[["study_hours","Attendance"]]
y=df["Result"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training samples:",len(x_train))
print("Training samples:",len(x_test))
Model=LogisticRegression()
Model.fit(x_train,y_train)