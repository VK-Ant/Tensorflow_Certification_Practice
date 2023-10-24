import tensorflow as tf
print(tf.__version__)
import datetime
print(datetime.datetime.now())

#create data and fit the model
import numpy as np
import matplotlib.pyplot as plt

#numpy array
X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

plt.scatter(X,y);
# plt.show()

#Example of input and output shapes of regression model
house_info = tf.constant(["Bedroom","Bathroom", "Garage"])
house_price = tf.constant([939700])
print(house_price,house_info)

print(house_info.shape)
print(house_price.shape)

#tensorflow
X = tf.constant([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
y = tf.constant([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

plt.scatter(X,y);
# plt.show();

#shape
input_shape = X[0].shape
output_shape = y[0].shape

print(input_shape)
print(output_shape)
print(X[0])
print(y[0])

#First model development

tf.random.set_seed(42)

model = tf.keras.Sequential([tf.keras.layers. Dense(1)])

#compile
model.compile(loss = tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])

#fit
model.fit(tf.expand_dims(X,axis=-1),y,epochs=5)

print(X,y)

#predict
pre = model.predict([17.0])
print(pre)
print("*************************************************************************************************")
#improve the model

tf.random.set_seed(42)

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

#compile
model.compile(loss = tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

#fit
model.fit(tf.expand_dims(X,axis=-1),y,epochs=100)

predict = model.predict([17.0])
print(predict)

#visualize

X1 = np.arange(-100,100, 4)
print(X1)

y1 = np.arange(-90,110,4)
print(y1)

print("length of X1:", len(X1))

X_train = X1[:40]
y_train = y1[:40]

X_test = X1[40:]
y_test = y1[40:]

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
# Plot test data in green
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show the legend
plt.legend();
# plt.show()

#build model
tf.random.set_seed(42)
model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=[1])])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
model.summary()

model.fit(X_train,y_train,epochs=100,verbose=0)
model.summary()

print("**************************************************")
# import tensorflow
# from tensorflow.keras.utils import plot_model
#
# plot_model(model, show_shape=True)
# plt.show()
print("**************************************************")
#predict
y_preds = model.predict(X_test)
print("Test_prediction:",y_preds)
print("**************************************************")

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
  plt.legend();
  plt.show()
print("**************************************************")

plot_predictions(train_data=X_train,
                   train_labels=y_train,
                   test_data=X_test,
                   test_labels=y_test,
                   predictions=y_preds)

print("**************************************************")

# Calculate the MSE
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=y_preds.squeeze())
print(mse)

print("**************************************************")

def mae(y_test, y_pred):
    """
    Calculuates mean absolute error between y_test and y_preds.
    """
    return tf.metrics.mean_absolute_error(y_test,
                                          y_pred)


def mse(y_test, y_pred):
    """
    Calculates mean squared error between y_test and y_preds.
    """
    return tf.metrics.mean_squared_error(y_test,
                                         y_pred)

print("**************************************************")

# Calculate model_1 metrics
mae_1 = mae(y_test, y_preds.squeeze()).numpy()
mse_1 = mse(y_test, y_preds.squeeze()).numpy()
print(mae_1, mse_1)
print("**************************************************")
# Set random seed
tf.random.set_seed(42)

# Replicate model_1 and add an extra layer
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)  # add a second layer
])

# Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model
model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)  # set verbose to 0 for less output

# Make and plot predictions for model_2
y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)

# Calculate model_2 metrics
mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()
print(mae_2, mse_2)
print("**************************************************")
#save the model
# model_2.save('best_model_saveedmodel_format')
model_2.save('bestmodel.h5')

#Load the model
load_saved_model = tf.keras.models.load_model("bestmodel.h5")
load_saved_model.summary()

#colab to download

# # Download the model (or any file) from Google Colab
# from google.colab import files
#
# files.download("best_model_HDF5_format.h5")

print("**************************************************")
#medical insurance
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read csv
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(insurance.head())
#turn all cat to num
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())

#x and y value split
X = insurance_one_hot.drop('charges',axis=1)
y = insurance_one_hot["charges"]

print(X.head())

#split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
print("-------------------------------------------------------------------")
#build the model

# Set random seed
tf.random.set_seed(42)

# Create a new model (same as model_2)
insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

# Fit the model
insurance_model.fit(X_train, y_train, epochs=100)

insurance_model.evaluate(X_test,y_test)