#create timestamp

import datetime

print(f"Notebook last run (end to end): {datetime.datetime.now()}")

#Import and version

import tensorflow as tf

print("Tensorflow Version:",tf.__version__)

#create a scalar

scalar = tf.constant(7)
print("scalar:",scalar)
print("check the dimension:", scalar.ndim)

#vector
vector = tf.constant([10,10])
print("vector:",vector)
print("vector dimension:",vector.ndim)

#Matrix
matrix = tf.constant([[10,7],
                      [7,10]])
print("Matrix:", matrix)
print("Matrix dimension:", matrix.ndim)

#create another matrix and define the datatype

another_matrix = tf.constant([[10,7],
                              [3,2],
                              [8,9]],dtype=tf.float16)
print("Another matrix:", another_matrix)
print("Another matrix dimension:", another_matrix.ndim)

#tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])

print("tensor:",tensor)
print("tensor dimension:", tensor.ndim)

#tf.constant -> immutable (cannot be changed)
#tf.variable -> mutable (changed)

changable_tensor = tf.Variable([10,7])
unchangable = tf.constant([10,7])


print('changeable and unchangle tensor:', changable_tensor,unchangable)

changable_tensor[1].assign(10)
print(changable_tensor)

# unchangable[1].assign(10)
# print(unchangable)

#same seed output is same
#tf.random.Generator

random1 = tf.random.Generator.from_seed(42)
random1 = random1.normal(shape=(2,2))
random2 = tf.random.Generator.from_seed(42)
random2 = random2.normal(shape=(2,2))

print(random1,random2,random2 == random2)

#different seed

random3 = tf.random.Generator.from_seed(42)
random3 = random3.normal(shape=(2,2))
random4 = tf.random.Generator.from_seed(11)
random4 = random4.normal(shape=(2,2))

print(random3,random4,random3 == random4)

#shuffle concept: Prevent overfiting
not_shuffle = tf.constant([[10,10],
                          [1,10],
                          [10,210]])
#shuffle
print(tf.random.shuffle(not_shuffle))

#if you want shuffle in same manner random seed is useful parameter
print(tf.random.shuffle(not_shuffle,seed=42))

#other ways to make tensor
o = tf.ones(shape=(3,3))
print(o)
z = tf.zeros(shape=(3,3))
print(z)

#numy
import numpy as np
numpy_A = np.arange(1,25, dtype=np.int32)
A = tf.constant(numpy_A,
                shape=[2,4,3])
print(numpy_A)
print("-------------------------------------")
print(A)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")
rank_4_tensor = tf.zeros([2,3,4,5])
print(rank_4_tensor)

print(rank_4_tensor.shape)
print(rank_4_tensor.ndim)
print(tf.size(rank_4_tensor))
print("-----------------------------------------------------------")

# Get various attributes of tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy())  # .numpy() converts to NumPy array
print("-----------------------------------------------------------")

print(rank_4_tensor[:2,:2,:2,:2])

#last item
print(rank_4_tensor[:, -1])

#add extra dimension
rank_4_tensor = rank_4_tensor[...,tf.newaxis]
print(rank_4_tensor.ndim)