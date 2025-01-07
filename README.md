# faces-with-mask
facemask repository
![image](https://github.com/user-attachments/assets/7f810827-66df-4798-bed1-11a709130039)

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator from
tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D from tensorflow.keras.layers import Dropout from tensorflow.keras.layers import Flatten from tensorflow.keras.layers import Dense from tensorflow.keras.layers import Input from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split from sklearn.metrics import classification_report from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
# data loading

data = []
labels = []
for category in CATEGORIES:
path
os.path.join(DIRECTORY, category)
for img in os.listdir(path):
img_path = os.path.join(path, img)
image = load_img(img_path, target_size=(224, 224)) image = img_to_array(image)
image = preprocess_input(image)
data.append(image) labels.append(category)
# encoding

# perform one-hot encoding on the labels lb = LabelBinarizer()
labels = lb.fit_transform(labels) labels = to_categorical(labels)
data = np.array(data, dtype="float32") labels = np.array(labels)
# Splitting of data into training and Testing
(trainx, testx, trainy, testY) = train_test_split(data, labels, test_size=0.25, stratify-labels, random_state=42)
(trainx, testx, trainy, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug =
ImageDataGenerator
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")


# construct the training image generator for data augmentation
aug =
ImageDataGenerator
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")

# compile our model print("[INFO] compiling model...")
opt = Adam(1r=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer-opt,
|_ metrics=["accuracy"])
# train the head of the network print("[INFO] training head...") H = model.fit(
aug.flow(trainx, trainy, batch_size=BS),
steps_per_epoch=len (train) // BS,
validation_data=(testx, testY),
validation_steps-len (test) // BS,
epochs=EPOCHS)

# compile our model print("[INFO] compiling model...")
opt = Adam(1r=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer-opt, metrics=["accuracy"])
# train the head of the network print("[INFO] training head...") H = model.fit(
aug.flow(trainx, trainy, batch_size=BS),
steps_per_epoch=len (train) // BS,
validation_data=(testx, testY),
validation_steps-len (test) // BS,
epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network.....")
predIdxs = model.predict(testx, batch_size=BS)

# for each image in the testing set we need to find the index of the # label with corresponding largest predicted probability 

predIdxs
=
np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report print(classification_report(testy.argmax(axis=1), predIdxs,
target_names-lb.classes_))
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plots

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss") plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss") plt.title("Training and Validation Loss")
plt.xlabel("Epoch #") plt.ylabel("Loss")
plt.legend(loc= "upper right")
plt.savefig("plot300loss.png")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label=”train_acc") plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc") plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc= "lower right")
plt.savefig("plot300acc.png")


prototxtPath
=
r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk maskNet = load_model("mask_detector.model")
# initialize the video stream
print("[INFO] starting video stream...") VS = VideoStream(src=0).start()

