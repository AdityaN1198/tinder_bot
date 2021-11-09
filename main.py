import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import PIL as pillow

def predict_rating(user_image):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_dir= 'testing_imgs/'
    file = img_dir + user_image
    img = cv.imread(file)
    img = cv.resize(img,(600,400),interpolation= cv.INTER_LINEAR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)
    print(faces)
    print(img.shape)
    for (x,y,w,h) in faces:
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = img[y-30:y+h+30,x-30:x+w+30]
        #cv.imshow('img', crop_img)
        cv.imwrite('croped.png',crop_img)
        #
        #cv.imshow('org',img)
        cv.waitKey(0)
        break

    model = tf.keras.models.load_model('resmod.h5')



    print(model.summary())

    img = 'croped.png'


    try:
        img = tf.keras.preprocessing.image.load_img(
            img, grayscale=False, color_mode="rgb", target_size=(240,240))

    except FileNotFoundError:
        img=None
        print('Face not detected clearly')
        return None

    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])

    prediction = model.predict(input_arr)

    os.remove('croped.png')
    return  (prediction[0][0])



print(predict_rating('chris_evans.jpg'))