import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
color_pal = sns.color_palette()
plt.style.use('ggplot')

df = pd.read_csv('D:\csvdata\matrix_tem_data.csv')
for i in range(len(df['Datetime'])):
    df['Datetime'][i]=datetime.datetime.fromtimestamp(df['Datetime'][i])
df = df.set_index('Datetime')
print(df.head())

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Temperature data trend')
plt.show()

#Train / Test Split
train = df.loc[df.index < datetime.datetime.fromtimestamp(1694200046)]
valid = df.loc[(df.index < datetime.datetime.fromtimestamp(1694213471)) & (df.index >= datetime.datetime.fromtimestamp(1694200046))]
test = df.loc[df.index >= datetime.datetime.fromtimestamp(1694213471)]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
valid.plot(ax=ax, label='Valid Set')
test.plot(ax=ax, label='Test Set')
ax.axvline(datetime.datetime.fromtimestamp(1694200046), color='black', ls='--')
ax.axvline(datetime.datetime.fromtimestamp(1694213471), color='black', ls='--')
ax.legend(['Training Set', 'Valid Set','Test Set'])
plt.show()

#Feature Creation
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    return df

df = create_features(df)

#Create our Model
train = create_features(train)
valid = create_features(valid)
test = create_features(test)

FEATURES = ['minute', 'hour']
TARGET = 'tem_data'

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = test[FEATURES]
y_val = test[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

#MLFLOW
#Define the model hyperparameters
params = {
    "base_score": 0.5, 
    "booster": "gbtree",    
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "objective": "reg:squarederror",
    "max_depth": 3,
    "learning_rate": 0.01,
    "eval_metric": 'rmse'
}
mlflow.xgboost.autolog()
reg = xgb.XGBRegressor(**params)
results1=reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100)

#Forecast on Test
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['tem_data']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Truth Data and Prediction')
plt.show()

ax = df.loc[(df.index > datetime.datetime.fromtimestamp(1694213471)) & (df.index < datetime.datetime.fromtimestamp(1694226925))]['tem_data'] \
    .plot(figsize=(15, 5), title='Result comparison')
df.loc[(df.index > datetime.datetime.fromtimestamp(1694213471)) & (df.index < datetime.datetime.fromtimestamp(1694226925))]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Temperature prediction")

#Score (RMSE)
score = np.sqrt(mean_squared_error(test['tem_data'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    mlflow.log_dict(results1.evals_result(), "saved_results.json")
    mlflow.log_metric("RMSE", score)
    mlflow.set_tag("Training Info", "Basic XGBOOST model for temperature data")
    signature = infer_signature(X_train, reg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=reg,
        artifact_path="temperature_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )