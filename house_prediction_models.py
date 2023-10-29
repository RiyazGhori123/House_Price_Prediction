import numpy as np
import pandas as pd
<<<<<<< HEAD

data=pd.read_csv("House_Price_Prediction/Banglore_dataset/Bengaluru_House_Data.csv")
=======
# reading the dataset
data=pd.read_csv("Banglore_dataset\Bengaluru_House_Data.csv")

>>>>>>> b3d9b18454e846cb919ceb273d4a08c44fef46ee
data.head()
# Describes the shape of the file
data.shape

data.info()

for column in data.columns:
  print(data[column].value_counts())
  print("*"*20)

data.isna().sum()
# Removing unnecessry columns
data.drop(columns=['area_type','availability','society','balcony'],inplace=True)

data.describe()

data.info()

data['location'].value_counts()

data['location'] = data['location'].fillna('Sarjapur Road')

data['size'].value_counts()

data['size']=data['size'].fillna('2 BHK')

data['bath'] =data['bath'].fillna(data['bath'].median())

data.info()

data['bhk']=data['size'].str.split().str.get(0).astype(int)

data[data.bhk >20]

data['total_sqft'].unique()

def convertRange(x):
  temp=x.split('-')
  if len(temp)==2:
    return (float(temp[0])+ float(temp[1]))/2
  try:
    return float(x)
  except:
    return None

data['total_sqft']=data['total_sqft'].apply(convertRange)

data.head()

data['price_per_sqft']=data['price']*100000 / data['total_sqft']

data['price_per_sqft']

data.describe()

data['location'].value_counts()

data['location']=data['location'].apply(lambda x: x.strip())
location_count=data['location'].value_counts()

location_count

location_count_less_10=location_count[location_count<=10]
location_count_less_10

data['location']=data['location'].apply(lambda x :'other' if x in location_count_less_10 else x)

data['location'].value_counts()

"""#Outlier Detection and Removal"""

data.describe()

(data['total_sqft']/data['bhk']).describe()

data=data[((data['total_sqft']/data['bhk'])>=300)]
data.describe()

data.shape

data.price_per_sqft.describe()

# Removing outliers
def remove_outliers_sqft(df):
  df_output=pd.DataFrame()
  for key,subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_sqft)

    st=np.std(subdf.price_per_sqft)

    gen_df=subdf[(subdf.price_per_sqft> (m-st))&(subdf.price_per_sqft<= (m+st))]
    df_output=pd.concat([df_output,gen_df],ignore_index=True)
  return df_output

data=remove_outliers_sqft(data)
data.describe()

def bhk_outlier_remover(df):
  exclude_indices =np.array([])
  for location,location_df in df.groupby('location'):
    bhk_stats={}
    for bhk , bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean': np.mean(bhk_df.price_per_sqft),
          'std' : np.std(bhk_df.price_per_sqft),
          'count':bhk_df.shape[0]
      }

    for bhk,bhk_df in location_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices,axis='index')

data=bhk_outlier_remover(data)

data.shape

data

data.drop(columns=['size','price_per_sqft'],inplace=True)

"""#Cleaned Data"""

data.head()

data.to_csv("Cleaned_data.csv")

X=data.drop(columns=['price'])
y=data['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)

"""#Applying Linear Regression"""

column_trans=make_column_transformer((OneHotEncoder(sparse=False),['location']),
                                     remainder='passthrough')

scaler=StandardScaler()

# Remove the normalize=True argument from LinearRegression
lr = LinearRegression()

pipe = make_pipeline(column_trans, scaler, lr)

pipe.fit(X_train, y_train)

y_pred_lr = pipe.predict(X_test)

r2_score(y_test, y_pred_lr)

"""#Applying Lasso"""

lasso=Lasso()

pipe=make_pipeline(column_trans,scaler,lasso)

pipe.fit(X_train,y_train)

y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)

"""#Applying Ridge"""

ridge=Ridge()

pipe=make_pipeline(column_trans,scaler,ridge)

pipe.fit(X_train,y_train)

y_pred_ridge=pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)

print("Np Regularization: ",r2_score(y_test,y_pred_lr))
print("Lasso: ",r2_score(y_test,y_pred_lasso))
print("Ridge: ",r2_score(y_test,y_pred_ridge))

import pickle

pickle.dump(pipe,open('RidgeModel.pkl','wb'))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Create a ColumnTransformer to handle categorical features
# In this example, 'location' is assumed to be a categorical feature
categorical_features = ['location']
numeric_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Numeric features
        ('cat', OneHotEncoder(), categorical_features)  # Categorical features
    ])

# Creating a pipeline that combines preprocessing and DecisionTreeRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and calculate R-squared
y_pred_dt = pipeline.predict(X_test)
r2_score_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree R-squared:", r2_score_dt)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Assuming 'location' is a categorical feature
categorical_features = ['location']
numeric_features = [col for col in X.columns if col not in categorical_features]

# Creating a ColumnTransformer to handle categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Numeric features
        ('cat', OneHotEncoder(), categorical_features)  # Categorical features
    ])

# Creating a pipeline that combines preprocessing and RandomForestRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and calculate R-squared
y_pred_rf = pipeline.predict(X_test)
r2_score_rf = r2_score(y_test, y_pred_rf)
print("Random Forest R-squared:", r2_score_rf)

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

# Creating a OneHotEncoder to convert categorical features to one-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Assuming 'location' is a categorical variable
categorical_features = ['location']

# Fit and transform the categorical features to one-hot encoding
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Convert to DMatrix with the encoded categorical features
dtrain = xgb.DMatrix(X_train_encoded, label=y_train)
dtest = xgb.DMatrix(X_test_encoded)

# Creating and configuring the XGBoost regressor
xgb_reg = xgb.XGBRegressor()

# Train the XGBoost regressor
xgb_reg.fit(X_train_encoded, y_train)

# Predict
y_pred_xgb = xgb_reg.predict(X_test_encoded)

# Calculate R-squared
r2_score_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost R-squared:", r2_score_xgb)

from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Define which columns are categorical and which are numerical
categorical_columns = ['location']  # Add more if needed
numeric_columns = [col for col in X.columns if col not in categorical_columns]

# Creating transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combining transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Creating a pipeline that includes preprocessing and SVR
svr = SVR(kernel='linear')
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('svr', svr)])

# Fit the model
model.fit(X_train, y_train)

# Predict and calculate R-squared
y_pred_svr = model.predict(X_test)
r2_score_svr = r2_score(y_test, y_pred_svr)
print("SVR R-squared:", r2_score_svr)

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter Plot for Linear Regression
plt.scatter(y_test, y_pred_lr, c='blue', label='Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (Linear Regression)')
plt.show()

# Scatter Plot for Lasso Regression
plt.scatter(y_test, y_pred_lasso, c='green', label='Lasso Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (Lasso Regression)')
plt.show()

# Scatter Plot for Ridge Regression
plt.scatter(y_test, y_pred_ridge, c='red', label='Ridge Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (Ridge Regression)')
plt.show()

# Scatter Plot for Decision Tree
plt.scatter(y_test, y_pred_dt, c='purple', label='Decision Tree')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (Decision Tree)')
plt.show()

# Scatter Plot for Random Forest
plt.scatter(y_test, y_pred_rf, c='orange', label='Random Forest')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (Random Forest)')
plt.show()

# Scatter Plot for XGBoost
plt.scatter(y_test, y_pred_xgb, c='magenta', label='XGBoost')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (XGBoost)')
plt.show()

# Scatter Plot for SVR
plt.scatter(y_test, y_pred_svr, c='cyan', label='SVR')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Scatter Plot of Actual vs. Predicted Prices (SVR)')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Residual Plot for Linear Regression
residuals_lr = y_test - y_pred_lr
sns.residplot(x=y_pred_lr, y=residuals_lr, color='blue', label='Linear Regression')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (Linear Regression)')
plt.show()

# Line Plot (Actual vs. Predicted) for Linear Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_lr, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Linear Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Lasso Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_lasso, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Lasso Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Ridge Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_ridge, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Ridge Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Decision Tree
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_dt, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Decision Tree')
plt.show()

# Line Plot (Actual vs. Predicted) for Random Forest
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_rf, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Random Forest')
plt.show()

# Line Plot (Actual vs. Predicted) for XGBoost
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_xgb, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - XGBoost')
plt.show()

# Line Plot (Actual vs. Predicted) for SVR
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_svr, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - SVR')
plt.show()

# Residual Plot for Lasso Regression
residuals_lasso = y_test - y_pred_lasso
sns.residplot(x=y_pred_lasso, y=residuals_lasso, color='green', label='Lasso Regression')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (Lasso Regression)')
plt.show()

# Residual Plot for Ridge Regression
residuals_ridge = y_test - y_pred_ridge
sns.residplot(x=y_pred_ridge, y=residuals_ridge, color='red', label='Ridge Regression')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (Ridge Regression)')
plt.show()

import matplotlib.pyplot as plt

# Decision Tree Residual Plot
residuals_dt = y_test - y_pred_dt
plt.scatter(y_pred_dt, residuals_dt, c='purple', label='Decision Tree')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (Decision Tree)')
plt.show()

import matplotlib.pyplot as plt

# Random Forest Residual Plot
residuals_rf = y_test - y_pred_rf
plt.scatter(y_pred_rf, residuals_rf, c='orange', label='Random Forest')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (Random Forest)')
plt.show()

import matplotlib.pyplot as plt

# XGBoost Residual Plot
residuals_xgb = y_test - y_pred_xgb
plt.scatter(y_pred_xgb, residuals_xgb, c='magenta', label='XGBoost')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (XGBoost)')
plt.show()

import matplotlib.pyplot as plt

# SVR Residual Plot
residuals_svr = y_test - y_pred_svr
plt.scatter(y_pred_svr, residuals_svr, c='cyan', label='SVR')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot (SVR)')
plt.show()

# Line Plot (Actual vs. Predicted) for Linear Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_lr, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Linear Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Lasso Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_lasso, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Lasso Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Ridge Regression
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_ridge, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Ridge Regression')
plt.show()

# Line Plot (Actual vs. Predicted) for Decision Tree
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_dt, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Decision Tree')
plt.show()

# Line Plot (Actual vs. Predicted) for Random Forest
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_rf, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - Random Forest')
plt.show()

# Line Plot (Actual vs. Predicted) for XGBoost
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_xgb, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - XGBoost')
plt.show()

# Line Plot (Actual vs. Predicted) for SVR
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_svr, label='Predicted Prices', color='green')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices (Line Plot) - SVR')
plt.show()



from tabulate import tabulate

# Create a list of models and their predictions
models = [lr, lasso, ridge, dt, rf, xgb_reg, svr]
model_names = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVR']
predictions = [y_pred_lr, y_pred_lasso, y_pred_ridge, y_pred_dt, y_pred_rf, y_pred_xgb, y_pred_svr]

# Initialize a list of dictionaries to store evaluation metrics
model_metrics = []

# Evaluate each model and populate the metrics
for i, model in enumerate(models):
    r2 = r2_score(y_test, predictions[i])
    mae = mean_absolute_error(y_test, predictions[i])
    mse = mean_squared_error(y_test, predictions[i])
    rmse = np.sqrt(mse)

    model_metrics.append({
        'Model': model_names[i],
        'R-squared (R2)': f'{r2:.3f}',
        'Mean Absolute Error (MAE)': f'{mae:.3f}',
        'Mean Squared Error (MSE)': f'{mse:.3f}',
        'Root Mean Squared Error (RMSE)': f'{rmse:.3f}'
    })

# Print the metrics table
table = tabulate(model_metrics, headers="keys", tablefmt="pretty")
print("\nModel Evaluation Metrics:")
print(table)

# Find the best model based on R-squared
best_model_idx = max(model_metrics, key=lambda x: float(x['R-squared (R2)']))
best_model_name = best_model_idx['Model']
best_r2 = best_model_idx['R-squared (R2)']

print(f"\nThe best model is {best_model_name} with R2: {best_r2}")

