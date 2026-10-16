import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st

st.title("🎙️ Noise Cancellation from noisereduce library")
st.write(
    "Neeche diye gaye button se audio record karein aur wind noise hatayein!"
)

# Streamlit ka built-in audio recorder (Cloud aur Mobile dono par perfectly chalta hai)
audio_file = st.audio_input("🔴 Yahan click karke recording shuru karein:")

if audio_file is not None:
  st.subheader("🔊 Original Noisy Audio:")
  st.audio(audio_file, format="audio/wav")

  if st.button("🧹 Noise Reduce Karein"):
    with st.spinner("⏳ Noise reduce ho rahi hai..."):
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
          "✅ Noise successfully reduce ho chuki hai aur 'output_cleaned.wav'"
          " ban gayi hai!"
      )

      # Cleaned audio play karein
      st.subheader("🎧 Cleaned Audio (Output):")
      st.audio(output_path, format="audio/wav")
