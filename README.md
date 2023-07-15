# Diabetics-Classification-using-LR-KNN-SVC-DT-RF
   LogisticRegression
   KNeighborsClassifier
   DecisionTreeClassifier
   RandomForestClassifier

 # Import all the requirements
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split,GridSearchCV
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

# SCIKIT LEARN PIPELINE (REDUCES MULTIPLE STEPS OUTPUT OF EACH STEP IS  THE INPUT TO THE NEXT)
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.svm import SVC
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.pipeline import Pipeline
  from sklearn.ensemble import RandomForestClassifier
  import joblib

#  Load the dataset (assuming you have a CSV file named 'diabetes_data.csv')
  data = pd.read_csv('diabetes.csv')

#  DISPLAYING FIRST FIVE ROWS OF THE DATASET
  data.head()

# CHECKING THE LAST FIVE ROWS OF THE DATASET
  data.tail()

# SHAPE OF OUR DATASET(TOTAL NO OF ROWS AND COLUMS)
  data.shape

# GET THE INFORMATION OF OUR DATASET
  data.info()

# CHECKING FOR NULL VALUES IN THE DATASET
  data.isnull()
  data.isnull().sum()

# OVERALL STATS OF OUR DATASET
  data.describe()

  data_copy = data.copy(deep=True)
  data.columns
  
  data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.nan)
  data_copy.isnull().sum()

  data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
  data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
  data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
  data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
  data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())

# STORE THE FEATURE MATRIX(INDEPENDENT VARIABLE) IN X AND  RESPONSE(TARGET) IN Y
  x = data.drop('Outcome',axis=1)
  y = data['Outcome']

  y

# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
  from sklearn.model_selection import train_test_split
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#  CREATING PIPELINE USING SKLEARN
  pipeline_lr = Pipeline([('scalar1',StandardScaler()),
  ('lr_classifier',LogisticRegression())])

  pipeline_knn = Pipeline([('scalar2',StandardScaler()),
  ('knn_classifier',KNeighborsClassifier())])

  pipeline_svc = Pipeline([('scalarr3',StandardScaler()),
  ('svc_classifier',SVC())])
  
  pipeline_dt = Pipeline([('dt_classifier',DecisionTreeClassifier())])
  pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier())])
  
  pipelines =[pipeline_lr,
   pipeline_knn,
   pipeline_svc,
   pipeline_dt,
   pipeline_rf]
  pipelines
  
  for pipe in pipelines:
   pipe.fit(x_train,y_train)
  pipe_dict = {0:'LR',
   1:'KNN',
   2:'SVC',
   3:'DT',
   4:'RF'}
  pipe_dict
  
  for i,model in enumerate(pipelines):
      print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(x_test,y_test)*100))
  
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier(max_depth=3)
  rf.fit(x,y)

# PREDICTION ON NEW DATA
  new_data = pd.DataFrame({
   'Pregnancies':6,
   'Glucose':148.0,
   'BloodPressure':72.0,
   'SkinThickness':35.0,
   'Insulin':79.799479,
   'BMI':33.6,
   'DiabetesPedigreeFunction':0.627,
   'Age':50,
  },index=[0])
  
  
  p=rf.predict(new_data)
  if p[0] == 0:
   print('non-diabetic')
  else:
   print('diabetic')

# SAVE MODEL USING JOBLIB
  import joblib
  joblib.dump(rf,'model_joblib_diabetes')
  
  model = joblib.load('model_joblib_diabetes')
  model.predict(new_data)


  


