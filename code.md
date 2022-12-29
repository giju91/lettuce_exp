# lettuce
master course graduation

#first of all, this code can fit on the google colaboratory
#so U want to use this code, I recommend U use it through google colaboratory

#upload
from google.colab import files
myfile = files.upload()

import io
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


data = pd.read_csv(io.BytesIO(myfile['csvfinal4.csv']))
data.head()

corr_matrix = data.corr()
corr_matrix["Anto"].sort_values(ascending=False)

corr_df = pd.DataFrame(corr_matrix["Anto"].sort_values(ascending=False))


plt.figure(figsize=(12,8))
plt.bar(corr_df.index, corr_df["Anto"])
plt.xticks(rotation=45)

#Visualization
attributes = ['Anto','Darkness Value','B','K=1,B','K=1,G', 'K=1,2G-R']
scatter_matrix(data[attributes], figsize=(12, 8))

#Train
from sklearn.model_selection import train_test_split
x = data[['Darkness Value','B','K=1,G', 'K=1,2G-R']]
y = data[['Anto']]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size = 0.3)

#Model fitting
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

#Score
mlr.coef_
mlr.intercept_
print(mlr.score(x_train, y_train))

#Visualization
import matplotlib.pyplot as plt
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual anthocyanin")
plt.ylabel("Predicted anthocyanin")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

