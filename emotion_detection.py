import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("emotion.hdf5")
cascade = cv2.CascadeClassifier("cascade_frontal_face.xml")
emotions = ('Anger','disgust','fear','Happy','sad','surprised','neutral')
cam = cv2.VideoCapture(0)

while True:
    image = cam.read()
    image = image[1]
    faces = cascade.detectMultiScale(image,minNeighbors=5)
    for (x,y,z,h) in faces:
        face_image = image[y:y+h,x:x+z]
        cv2.rectangle(image,(x,y),(x+z,y+h),(255,0,0),3)
        face_image = cv2.cvtColor(face_image,cv2.COLOR_RGB2GRAY)
        face_image = cv2.resize(face_image,(48,48))
        face_image = face_image.astype("float")/255.0
        print(face_image.shape)
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image,axis=0)
        result = model.predict(face_image)[0]
        result_index = np.argmax(result)
        predicted_emotion = emotions[result_index]
        print(predicted_emotion)
        cv2.putText(image,predicted_emotion,(x,y),cv2.FONT_ITALIC,1,(255,0,0),2)
    cv2.imshow("result",image)
    if cv2.waitKey(25) & 0xFF==ord("q"):
        break
cv2.destroyAllWindows()
