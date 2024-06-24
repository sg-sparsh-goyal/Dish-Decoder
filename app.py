import alt as alt
import streamlit as st
import pandas as pd
import tensorflow as tf
import altair as alt
from utils import load_and_prep, get_classes, preprocess_data  # Import the preprocess_data function
import time

# @st.cache_data(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df


class_names = get_classes()

st.set_page_config(page_title="Dish Decoder",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Dish Decoder ?")
st.sidebar.write("""
Dish Decoder is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 

- It can identify over 100 different food classes

- It is based upon a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

- The Model actually beats the DeepFood Paper's model which also trained on the same dataset.

- The Accuracy acquired by DeepFood was 77.4% and our model's 85%.

- Difference of 8% ain't much, but the interesting thing is, DeepFood's model took 2-3 days to train while this barely took 90min.

**Accuracy :** **`85%`**

**Model :** **`EfficientNetB1`**

**Dataset :** **`Food101`**

""")

#### Main Body ####

st.title("Dish Decoder 🍔👁️")
st.header("Discover, Decode, Delight !")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])

model = tf.keras.models.load_model("FoodVision.hdf5")

st.sidebar.markdown("Created by **Sparsh Goyal**")

st.markdown(
    """
    <div style="position: fixed; bottom: 0; right: 10px; padding: 10px; color: white;">
        <a href="https://github.com/sg-sparsh-goyal" target="_blank" style="color: white; text-decoration: none;">
            ✨ Github
        </a><br>
    </div>
    """,
    unsafe_allow_html=True
)

if not file:
    st.warning("Please upload an image")
    st.stop()
else:
    st.info("Uploading your image...")

    # Add a loading bar
    progress_bar = st.progress(0)
    image = file.read()

    # Simulate image processing with a 2-second delay
    for percent_complete in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent_complete + 1)

    st.success("Image upload complete!")

    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf * 100:.2f}%')
    chart = alt.Chart(df).mark_bar(color='#00FF00').encode(
        x=alt.X('F1 Scores', axis=alt.Axis(title=None)),
        y=alt.Y('Top 5 Predictions', sort=None, axis=alt.Axis(title=None)),
        text='F1 Scores'
    ).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)
