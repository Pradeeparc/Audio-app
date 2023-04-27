import streamlit as stl
import librosa
import numpy as np
import tensorflow as tf



model_load = tf.keras.models.load_model("audio_classification.hdf5")
#print(list(model_load.keys()))

stl.title("Audio_Classification")
Audio = stl.file_uploader("Upload file",type=["mp3","wav"])
if Audio is not None:
    stl.audio(Audio)

btn = stl.button("PREDICT")
if btn == True:
    y=["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "enginge_idling", "gun_shot", "jackhammer", "siren","street_music"]
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    y=to_categorical(labelencoder.fit_transform(y))

    filename=Audio
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label=np.argmax(model_load.predict(mfccs_scaled_features),axis=1)
    print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    print(prediction_class)

    import streamlit as st2
    # st2.write(prediction_class)

    if prediction_class == "dog_bark":
        st2.header(" Audio content identified as: DOG BARK ")
    elif prediction_class == "air_conditioner":
        st2.header(" Audio predicted as : AIR CONDITIONER ")
    elif prediction_class == "car_horn":
        st2.header("Audio predicted as : CAR HORN ")
    elif prediction_class == "children_playing":
        st2.header("Audio predicted as : CHILDREN PLAYING ")
    elif prediction_class == "drilling":
        st2.header("Audio predicted as : DRILLING ")
    elif prediction_class == "enginge_idling":
        st2.header("Audio predicted as : ENGINE IDLING ")
    elif prediction_class == "gun_shot":
        st2.header("Audio predicted as : GUN SHOT ")
    elif prediction_class == "jackhammer":
        st2.header("Audio predicted as : JACKHAMMER ")
    elif prediction_class == "siren":
        st2.header("Audio predicted as : SIREN ")
    elif prediction_class == "street_music":
        st2.header("Audio predicted as : STREET MUSIC")
    
    
