import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str, target_num_frames: int = 75) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust the number of frames
    frame_indices = [int(i * total_frames / target_num_frames) for i in range(target_num_frames)]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        # Check if frame is not None
        if ret and frame is not None:
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236, 80:220, :])

    cap.release()

    if not frames:
        raise ValueError("No frames were loaded from the video.")

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join(r"C:\Users\harin\OneDrive\Desktop\LIPNET\LipNet-main\data\s1", f'{file_name}.mpg')
    alignment_path = os.path.join(r"C:\Users\harin\OneDrive\Desktop\LIPNET\LipNet-main\data\alignments\s1", f'{file_name}.align')
    frames = load_video(video_path, target_num_frames=75)
    alignments = load_alignments(alignment_path)

    return frames, alignments