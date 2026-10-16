import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st

st.title("🎙️ Noise Cancellation from noisereduce library")
st.write(
    "record your voice from this botton!"
)

# Streamlit ka built-in audio recorder (Cloud aur Mobile dono par perfectly chalta hai)
audio_file = st.audio_input("🔴 click here to record your voice :")

if audio_file is not None:
  st.subheader("🔊 Original Noisy Audio:")
  st.audio(audio_file, format="audio/wav")

  if st.button("🧹 reduce your voice "):
    with st.spinner("⏳ Noise reducing..."):
      # Audio data read karein
      data, rate = sf.read(audio_file)

      # Agar stereo ho toh mono mein convert karein
      if len(data.shape) > 1:
        data = np.mean(data, axis=1)

      # Noise reduce apply karein
      reduced_audio = nr.reduce_noise(y=data, sr=rate)

      # Clean audio save karein
      output_path = "output_cleaned.wav"
      sf.write(output_path, reduced_audio, rate)

      st.success(
          "✅ Noise successfully reduced 'output_cleaned.wav'"
          " updated!"
      )

      # Cleaned audio play karein
      st.subheader("🎧 Cleaned Audio (Output):")
      st.audio(output_path, format="audio/wav")
