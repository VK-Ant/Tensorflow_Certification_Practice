#Manipulating tensor (tensor operation)

import tensorflow as tf
print(tf.__version__)

#Basic Operation

tensor = tf.constant([[10,7],[3,4]])
print("Addition:", tensor + 20)
print('*********************************')
print("Sub:", tensor - 20)
print('*********************************')
print("Multiply:", tensor * 20)
print('*********************************')
print("Divide:", tensor / 20)
print('*********************************')
print("Multiply:",tf.multiply(tensor, 10))

print('*********************************')

print("input",tensor)
print("Matrix multiply",tf.matmul(tensor,tensor))
print("Matrix multiply",tensor @ tensor)
print('*********************************')

#reshape

# Create (3, 2) tensor
X = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])

# Create another (3, 2) tensor
Y = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])
print(X,Y)
# Example of reshape (3, 2) -> (2, 3)
print("reshape:",tf.reshape(Y, shape=(2, 3)))

print("Multiply after reshape the data:",X @ tf.reshape(Y, shape=(2, 3)))
print('*********************************')

#transpose
print(tf.transpose(X))
# You can achieve the same result with parameters
y = tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False)
print(y)
print('*********************************')

#Dot product
print(tf.tensordot(tf.transpose(X),Y, axes=1))
print('*********************************')
#change the datatype of a tensor using tf.cast().


# Create a new tensor with default datatype (float32)
B = tf.constant([1.7, 7.4])

# Create a new tensor with default datatype (int32)
C = tf.constant([1, 7])
print(B,C)
print('*********************************')
print(tf.cast(B,dtype=tf.float16))
print('*********************************')
#absolute values
# Create tensor with negative values
D = tf.constant([-7, -10])
print(D)
print(tf.abs(D))
print('*********************************')
#Min,Max,Mean, sum,argument max,min
import numpy as np
# Create a tensor with 50 random values between 0 and 100
E = tf.constant(np.random.randint(low=0, high=100, size=50))
print(E)
print(tf.reduce_min(E))
print(tf.reduce_max(E))
print(tf.reduce_mean(E))
print(tf.reduce_sum(E))
print(tf.argmax(E))
print(tf.argmin(E))

print('*********************************')

#squeezing a tensor (single dimension)

# Create a rank 5 (5 dimensions) tensor of 50 numbers between 0 and 100
G = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
print(G.shape, G.ndim)


# Squeeze tensor G (remove all 1 dimensions)
G_squeezed = tf.squeeze(G)
print(G_squeezed.shape, G_squeezed.ndim)
print('*********************************')
#one hot encoding

# Create a list of indices
some_list = [0, 1, 2, 3]
oh = tf.one_hot(some_list, depth=4)
print(oh)
oh_1=tf.one_hot(some_list, depth=4, on_value="Hi vk!", off_value="back to form kaggler")
print(oh_1)
print('*********************************')
H = tf.constant(np.arange(1, 10))
H = tf.cast(H, dtype=tf.float32)

print(tf.square(H))
print(tf.sqrt(H))
print(tf.math.log(H))
print('*********************************')
I = tf.Variable(np.arange(0, 5))
print(I)
print(I.assign([0, 1, 2, 3, 50]))
print(I.assign_add([10, 10, 10, 10, 10]))
print('*********************************')
#decorators
# Create a simple function
def function(x, y):
  return x ** 2 + y

x = tf.constant(np.arange(0, 10))
y = tf.constant(np.arange(10, 20))
print(function(x, y))


# Create the same function and decorate it with tf.function
@tf.function
def tf_function(x, y):
  return x ** 2 + y

print(tf_function(x, y))

print('*********************************')
#find gpu access
print(tf.config.list_physical_devices('GPU'))
