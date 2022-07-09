import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
lr_model=LogisticRegression(n_jobs=-1)
lr_model.fit(X_train,y_train)
rfc=RandomForestClassifier(n_jobs=-1,n_estimators=100)
rfc.fit(X_train,y_train)
@st.cache()
def predict_species(a,b,c,d,model):
  predicted=model.predict([[a,b,c,d]])
  predicted=predicted[0]
  if predicted==0:
    return 'Iris-setosa'
  elif predicted==1:
    return "Iris-virginica"
  elif predicted==2:
    return "Iris-versicolor"  
st.sidebar.title("Predicting the species for Iris flowers")   
aa=st.sidebar.slider("Sepal length",float(iris_df["SepalLengthCm"].min()),float(iris_df["SepalLengthCm"].max()))   
bb=st.sidebar.slider("Sepal width",float(iris_df["SepalWidthCm"].min()),float(iris_df["SepalWidthCm"].max()))   
cc=st.sidebar.slider("Petal length",float(iris_df["PetalLengthCm"].min()),float(iris_df["PetalLengthCm"].max()))   
dd=st.sidebar.slider("Petal width",float(iris_df["PetalWidthCm"].min()),float(iris_df["PetalWidthCm"].max())) 
model_name=st.sidebar.selectbox("Select the model",("Support vector Machine","Logistic Regression","Random Forest Classifier"))  
if st.sidebar.button("Classify"):
    if model_name=="Random Forest Classifier":
      ps=predict_species(aa,bb,cc,dd,rfc)
      score=rfc.score(X_train,y_train)
    elif model_name=="Support vector Machine"  :
      ps=predict_species(aa,bb,cc,dd,svc_model)
      score=svc_model.score(X_train,y_train)
    elif model_name=="Logistic Regression":
      ps=predict_species(aa,bb,cc,dd,lr_model)
      score=lr_model.score(X_train,y_train)
    st.write("The species is",ps)
    st.write("The score of the model is",score)
         
    
 
  