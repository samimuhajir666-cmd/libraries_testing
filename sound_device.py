import numpy as np
import soundfile as sf
import noisereduce as nr
import streamlit as st

st.title("🎙️ noice cancelation from noice reduce library")
st.write(
    "start your recording from given botton "
    " remove loudness!"
)

# 1. Recording duration set karein
duration = 20 # Seconds
sample_rate = 44100

if st.button("🔴 Start Recording "):
  with st.spinner(
      "🎙️ recording start... !"
  ):
    # Mic se audio record karein
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    # File save karein
    sf.write("sample_noisy.wav", audio_data, sample_rate)
    st.success("✅ Recording completed  'sample_noisy.wav' build !")

    # Original audio play karein
    st.subheader("🔊 Original Noisy Audio:")
    st.audio("sample_noisy.wav", format="audio/wav")

    # Noise reduce karein
    with st.spinner("🧹 Noise reduce ho rahi hai..."):
      reduced_audio = nr.reduce_noise(y=audio_data.flatten(), sr=sample_rate)
      sf.write("output_cleaned.wav", reduced_audio, sample_rate)

    # Cleaned audio play karein
    st.subheader("🎧 Cleaned Audio (Output):")
    st.audio("output_cleaned.wav", format="audio/wav")
