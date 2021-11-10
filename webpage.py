import streamlit as st
from PIL import Image
import cv2 as cv
from main import predict_rating
import numpy as np


st.title('Image Rating App')

st.write('This is a webapp that predicts your attractiveness rating. You can upload your image in .jpg .png .jpeg '
         'Rating is out of 5. Dont get upset if you get a low rating, the AI is not perfect, neither are you.'
         'Before you use it, I want to say is there is more to a human than physical attraction. \n'
         'Looks are not everything..... but they are something.')

st.write('Tip: Try to upload a front facing photo for better results.')

with st.form("my-form", clear_on_submit=True):
    img = st.file_uploader(label='Your Image',type=['png','jpg','jpeg'],accept_multiple_files=False)
    submitted = st.form_submit_button("UPLOAD!")


if img:
    img = Image.open(img)
    img = np.array(img.convert('RGB'))
    img = cv.cvtColor(img,0)
    st.image(img)
    rating = (predict_rating(img))
    rating = float(rating)
    if type(rating) is float:
        st.markdown('## Your Rating is {:.2f} out of 5'.format(rating))

    else:
        st.markdown('## {}'.format(rating))
