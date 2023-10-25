import tensorflow as tf
from helper1 import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

print(f"TensorFlow version: {tf.__version__}")


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def create_base_model(input_shape: tuple[int, int, int] = (224, 224, 3),
                      output_shape: int = 10,
                      learning_rate: float = 0.001,
                      training: bool = False) -> tf.keras.Model:
    """
    Create a model based on EfficientNetV2B0 with built-in data augmentation.

    Parameters:
    - input_shape (tuple): Expected shape of input images. Default is (224, 224, 3).
    - output_shape (int): Number of classes for the output layer. Default is 10.
    - learning_rate (float): Learning rate for the Adam optimizer. Default is 0.001.
    - training (bool): Whether the base model is trainable. Default is False.

    Returns:
    - tf.keras.Model: The compiled model with specified input and output settings.
    """

    # Create base model
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
    base_model.trainable = training

    # Setup model input and outputs with data augmentation built-in
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  # pass augmented images to base model but keep it in inference mode
    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = layers.Dense(units=output_shape, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    return model


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

print("******************************************************")

walk_through_dir("10_food_classes_10_percent")

# Create training and test directories
train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

#preprocessing
IMG_SIZE = (224, 224)  # define image size
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            # what type are the labels?
                                                                            batch_size=32)  # batch_size is 32 by default, this is generally a good number
test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="categorical")

# Check out the class names of our dataset
print(train_data_10_percent.class_names)
print("------------------------------------")
# #Model 0: Building a transfer learning model using the Keras Functional API
# # 1. Create base model with tf.keras.applications
# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
#
# # OLD
# # base_model = tf.keras.applications.EfficientNetB0(include_top=False)
#
# # 2. Freeze the base model (so the pre-learned patterns remain)
# base_model.trainable = False
#
# # 3. Create inputs into the base model
# inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
#
# # 4. If using ResNet50V2, add this to speed up convergence, remove for EfficientNetV2
# # x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
#
# # 5. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNetV2 inputs don't have to be normalized)
# x = base_model(inputs)
# # Check data shape after passing it to base_model
# print(f"Shape after base_model: {x.shape}")
#
# # 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
# x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
# print(f"After GlobalAveragePooling2D(): {x.shape}")
#
# # 7. Create the output activation layer
# outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)
#
# # 8. Combine the inputs with the outputs into a model
# model_0 = tf.keras.Model(inputs, outputs)
#
# # 9. Compile the model
# model_0.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=["accuracy"])
#
# # 10. Fit the model (we use less steps for validation so it's faster)
# history_10_percent = model_0.fit(train_data_10_percent,
#                                  epochs=5,
#                                  steps_per_epoch=len(train_data_10_percent),
#                                  validation_data=test_data_10_percent,
#                                  # Go through less of the validation data so epochs are faster (we want faster experiments!)
#                                  validation_steps=int(0.25 * len(test_data_10_percent)),
#                                  # Track our model's training logs for visualization later
#                                  callbacks=[
#                                      create_tensorboard_callback("transfer_learning", "10_percent_feature_extract")])

print("***********************************************************")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Create a functional model with data augmentation
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.experimental import preprocessing # OLD

# NEW: Newer versions of TensorFlow (2.10+) can use the tensorflow.keras.layers API directly for data augmentation
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNet
], name ="data_augmentation")

## OLD
# # Build data augmentation layer
# data_augmentation = Sequential([
#   preprocessing.RandomFlip('horizontal'),
#   preprocessing.RandomHeight(0.2),
#   preprocessing.RandomWidth(0.2),
#   preprocessing.RandomZoom(0.2),
#   preprocessing.RandomRotation(0.2),
#   # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNet
# ], name="data_augmentation")

# # Setup the input shape to our model
# input_shape = (224, 224, 3)
#
# # Create a frozen base model
# # base_model = tf.keras.applications.EfficientNetB0(include_top=False)
# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
# base_model.trainable = False
#
# # Create input and output layers
# inputs = layers.Input(shape=input_shape, name="input_layer") # create input layer
# x = data_augmentation(inputs) # augment our training images
# x = base_model(x, training=False) # pass augmented images to base model but keep it in inference mode, so batchnorm layers don't get updated: https://keras.io/guides/transfer_learning/#build-a-model
# x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
# outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
# model_2 = tf.keras.Model(inputs, outputs)
#
# # Compile
# model_2.compile(loss="categorical_crossentropy",
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # use Adam optimizer with base learning rate
#               metrics=["accuracy"])


# Create an instance of model_2 with our new function
model_2 = create_base_model()
# Setup checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"  # note: remember saving directly to Colab is temporary

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         # set to False to save the entire model
                                                         save_best_only=True,
                                                         # save only the best model weights instead of a model every epoch
                                                         save_freq="epoch",  # save every epoch
                                                         verbose=1)
# Fit the model saving checkpoints every epoch
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_data,
                                          validation_steps=int(0.25 * len(test_data)),
                                          # do less steps per validation (quicker)
                                          callbacks=[
                                              create_tensorboard_callback("transfer_learning", "10_percent_data_aug"),
                                              checkpoint_callback])

# Evaluate on the test data
results_10_percent_data_aug = model_2.evaluate(test_data)
print(results_10_percent_data_aug)

# Plot model loss curves
plot_loss_curves(history_10_percent_data_aug)