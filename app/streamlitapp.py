# Import all of the dependencies
import streamlit as st
import os
import numpy as np
from PIL import Image
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import zipfile
import subprocess

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipNet')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip Reading AI')
# Generating a list of options or videos
base_path = r"C:\Users\harin\OneDrive\Desktop\LIPNET\LipNet-main\data\s1"
options = os.listdir(base_path)
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

# ... (previous code)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(base_path, selected_video)
        output_video_path = os.path.join(base_path, 'selected_video.mp4')  # Output video path

        # Provide the full path to the ffmpeg executable
        ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # Replace with the actual path to ffmpeg.exe
        conversion_command = '{} -i "{}" -vcodec libx264 "{}" -y'.format(ffmpeg_path, file_path, output_video_path)

        # Run the conversion command using subprocess
        try:
            subprocess.run(conversion_command, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Error converting or loading the video. Please check the file path and try again.")
            st.error(f"FFmpeg command error: {e.stderr.decode()}")
            st.error(f"FFmpeg command: {conversion_command}")
            st.stop()

        if os.path.exists(output_video_path):
            with open(output_video_path, 'rb') as video:
                video_bytes = video.read()
                st.video(video_bytes)
        else:
            st.error(f"Error loading the converted video. Please check the file path and try again.")

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        frames, annotations = load_data(file_path)

        # Convert video frames to RGB format
        video_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(frames)).numpy().astype('float32')

        # Normalize the video frames to the range [0, 1]
        video_rgb = (video_rgb - video_rgb.min()) / (video_rgb.max() - video_rgb.min())

        # Ensure the video frames have a compatible data type
        video_np = (video_rgb * 255).astype('uint8')

        imageio.mimsave('animation.gif', [Image.fromarray(frame) for frame in video_np], fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()

        # Specify the path to the ZIP file
        zip_file_path = r'C:\Users\harin\OneDrive\Desktop\LIPNET\LipNet-main\models - checkpoint 96.zip.zip'

        # Specify the directory to extract the contents
        extraction_dir = r'C:\Users\Vinay\OneDrive\Desktop\lipreading\models'

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)

        # Specify the directory containing the extracted checkpoint files
        checkpoint_dir = r'C:\Users\Vinay\OneDrive\Desktop\lipreading\models'

        # Load the latest checkpoint path
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        # Check if the checkpoint path is found
        if checkpoint_path:
            try:
                model.load_weights(checkpoint_path)
                st.success(f"Successfully loaded checkpoint from {checkpoint_path}")
            except tf.errors.NotFoundError as e:
                st.error(f"Error loading checkpoint weights: {e}")
                st.stop()
        else:
            st.error(f"No checkpoint files found in {checkpoint_dir}")
            st.stop()

        yhat = model.predict(tf.expand_dims(tf.convert_to_tensor(frames), axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)