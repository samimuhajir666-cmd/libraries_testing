from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st

st.title("🎙️ Auto Noise Reduction App")
st.write(
    "Record audio, and noise reduction is applied automatically right away!"
)

# Create a folder for recordings
os.makedirs("saved_recordings", exist_ok=True)

# Streamlit audio recorder
audio_file = st.audio_input("🔴 Click here to record audio:")

if audio_file is not None:
  # Create a unique timestamp
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  noisy_filename = f"saved_recordings/noisy_{timestamp}.wav"
  cleaned_filename = f"saved_recordings/cleaned_{timestamp}.wav"

  # Read audio data
  data, rate = sf.read(audio_file)

  if len(data.shape) > 1:
    data = np.mean(data, axis=1)

  # 1. Save original noisy audio
  sf.write(noisy_filename, data, rate)

  # 2. Automatically apply noise reduction immediately
  with st.spinner("⏳ Automatically reducing noise..."):
    reduced_audio = nr.reduce_noise(y=data, sr=rate)
    sf.write(cleaned_filename, reduced_audio, rate)

  st.success("✨ Recording saved and noise reduction applied automatically!")

  # Display latest recording side-by-side
  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### 🔊 Latest Original")
    st.audio(noisy_filename, format="audio/wav")
    with open(noisy_filename, "rb") as f:
      st.download_button(
          label="📥 Download Noisy",
          data=f,
          file_name=f"noisy_{timestamp}.wav",
          mime="audio/wav",
          key=f"latest_noisy_{timestamp}",
      )

  with col2:
    st.markdown("### 🎧 Latest Cleaned")
    st.audio(cleaned_filename, format="audio/wav")
    with open(cleaned_filename, "rb") as f:
      st.download_button(
          label="📥 Download Cleaned",
          data=f,
          file_name=f"cleaned_{timestamp}.wav",
          mime="audio/wav",
          key=f"latest_cleaned_{timestamp}",
      )

# --- Organized History Section ---
st.markdown("---")
st.subheader("📂 Past Recording Sessions")

all_files = os.listdir("saved_recordings")
timestamps = sorted(
    [
        f.replace("noisy_", "").replace(".wav", "")
        for f in all_files
        if f.startswith("noisy_")
    ],
    reverse=True,
)

if timestamps:
  selected_session = st.selectbox(
      "Select a past session:",
      timestamps,
      format_func=lambda x: f"Session: {x[:4]}-{x[4:6]}-{x[6:8]} at {x[9:11]}:{x[11:13]}:{x[13:15]}",
  )

  if selected_session:
    noisy_path = f"saved_recordings/noisy_{selected_session}.wav"
    cleaned_path = f"saved_recordings/cleaned_{selected_session}.wav"

    col1, col2 = st.columns(2)

    with col1:
      st.markdown("### 🔊 Original Noisy")
      if os.path.exists(noisy_path):
        st.audio(noisy_path, format="audio/wav")
        with open(noisy_path, "rb") as f:
          st.download_button(
              label="📥 Download Noisy",
              data=f,
              file_name=f"noisy_{selected_session}.wav",
              mime="audio/wav",
              key=f"hist_noisy_{selected_session}",
          )

    with col2:
      st.markdown("### 🎧 Cleaned Audio")
      if os.path.exists(cleaned_path):
        st.audio(cleaned_path, format="audio/wav")
        with open(cleaned_path, "rb") as f:
          st.download_button(
              label="📥 Download Cleaned",
              data=f,
              file_name=f"cleaned_{selected_session}.wav",
              mime="audio/wav",
              key=f"hist_cleaned_{selected_session}",
          )
else:
  st.info("No recordings saved yet.")
