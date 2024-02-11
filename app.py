from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st
import numpy as np

def loaded_model():
    model=tf.keras.models.load_model("Transfer_Learning_Model.hdf5")
    return model

new_model=loaded_model()
st.set_page_config(page_title="Animal Classifier App", page_icon="🐾")
st.title("Animal Classifier App 🦁")
file = st.file_uploader("Please upload images of any the following animals: Elephant, Lion, Monkey, Zebra",type=["jpg", "png"])
def import_and_predict(image_data, new_model):
  size = (160,160)
  image = ImageOps.fit(image_data, size, method=0, bleed=0.0, centering=(0.5, 0.5))
  image = np.asarray(image)
  image = (image.astype(np.float32) / 255.0)  #NEW ADDED LINE
  img_reshape = image[np.newaxis,...]
  prediction = new_model.predict(img_reshape)
  return prediction

if file is not None:
  image = Image.open(file)
  predictions = import_and_predict(image, new_model)
  class_names = ['Elephant', 'Lion', 'Monkey','Zebra']
  string="The identified animal is: "+class_names[np.argmax(predictions)]
  st.success(string)
  st.image(image, use_column_width=True)

st.sidebar.markdown("<h2><u>OVERVIEW</u></h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "The Animal Classifier App utilizes a convolutional neural network (CNN) model "
    "trained on a diverse dataset of animal images. It has been fine-tuned to achieve "
    "a high accuracy of 96.25%.\n  Please note that while the model strives for"
    " accuracy, predictions may not always be correct.",
    unsafe_allow_html=True
)







