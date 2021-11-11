import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import logging
import tempfile
tf.get_logger().setLevel(logging.ERROR)
from PIL import Image

def predict_rating(user_image,read_from= None,debugging='off'):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    if read_from == 'disk':
        img_dir= 'testing_imgs/'
        file = img_dir+user_image
        img = cv.imread(file)
    else:
        img = user_image
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)



    #img = cv.resize(img,(600,400),interpolation= cv.INTER_LINEAR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    #print(faces)
    #print(img.shape)
    for (x,y,w,h) in faces:
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_size = 50
        while crop_size>0:
            try:
                crop_img = img[y-crop_size:y+h+crop_size,x-crop_size:x+w+crop_size]
                #cv.imwrite('croped.png',crop_img)
                break
            except cv.error:
                crop_size -= 10

        if debugging == 'on':
            cv.imshow('img', crop_img)
            #cv.imshow('org',img)
        cv.waitKey(0)
        break


    model = tf.keras.models.load_model('resmod.h5')



    #print(model.summary())

    #img = 'croped.png'

    try:
        crop_img = cv.cvtColor(crop_img,0)
        rgb = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    except UnboundLocalError:
        return 'Face not Detected in Photo'



    # try:
    #     pass
    #     #img = tf.keras.preprocessing.image.load_img(
    #         #img, grayscale=False, color_mode="rgb", target_size=(240,240))
    #
    # except FileNotFoundError or cv.error:
    #     img=None
    #
    #     return 'Face not detected clearly'
    #
    # input_arr = tf.keras.preprocessing.image.img_to_array(img)
    # input_arr = np.array([input_arr])

    prediction = model.predict(rgb_tensor)

    #os.remove('croped.png')
    return  (prediction[0][0])

#print(predict_rating('tyagi2.jpeg',read_from='disk',debugging='off'))
