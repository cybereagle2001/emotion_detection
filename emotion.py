import pandas as pd
import numpy as np
from arch import simple_CNN
from sklearn.model_selection import train_test_split

classes = ('Anger','disgust','fear','Happy','sad','surprised','neutral')

def load_data(path_data_used):
    data = pd.read_csv(path_data_used)
    pixels = data["pixels"].tolist()
    faces = []
    for p in pixels:
        #print("\npixels:\n",p)
        face = [int(pixel) for pixel in p.split(" ")]
        face = np.asarray(face).reshape(48,48)
        #print("np.asarray:",face)
        #second method
        #face = cv2.resize(face.astype("uint8"),(48,48))
        #print("CV2: ",face)
        faces.append(face)
    faces = np.expand_dims(faces,-1)
    emotions = pd.get_dummies(data["emotion"])
    #print("emotion:\n", emotions)
    return (faces,emotions)

def pre_processing(x):
    x = x.astype("float32")
    x = x/255.0
    return x

faces,emotions = load_data("fer2013.csv")
faces = pre_processing(faces) # in order to assure that numbers are between (0,1)
model = simple_CNN((48,48,1),7) # use of simple_CNN model from arch.py
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics="accuracy") # paramters (to identify the use of adam)
model.summary() # summarize the model architecture
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2) # split data into train and test data
model_CNN = model.fit(x_train,y_train,batch_size=64,epochs=5,validation_data=(x_test,y_test),verbose=1) # model training
model.save("emotion.hdf5") # save the model into emotion.hdf5
