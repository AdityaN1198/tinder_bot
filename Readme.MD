# Image Rating App

This is a DL bot that rates your image out of 5. The app is hosted on streamlit's platform which is free for a month. In
case it is not working, then the service has expired. You can try to replicate it locally by cloning the repo.

The app takes around 5500 images, rated by **South China University of Technology**. The image dataset is available on 
their github https://github.com/HCIILAB

**Resent50** is the model used for training with only top layer as trainable. It also uses a **openCV Cascade Model** to detect 
faces from image for better rating.

Following are the attached screenshot of the app

![Image of app](./Screenshot%202021-12-02%20at%2019-20-32%20Streamlit.png "Image of the App hosted on streamlit")

![Image of app](./Screenshot%202021-12-02%20at%2019-21-26%20Streamlit.png)

PS- The repo is called tinder bot because the initial idea was to use it for automating tinder on basis of rating but 
due to verification problems I decide to change it to different direction.

