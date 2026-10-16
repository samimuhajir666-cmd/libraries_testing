from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st

st.title("🎙️ Multi-Recording Noise Reduction App")
st.write(
    "Record audio, clean background noise, and download your files instantly!"
)

# Create a folder for recordings
os.makedirs("saved_recordings", exist_ok=True)

# Streamlit audio recorder
audio_file = st.audio_input("🔴 Click here to start a new recording:")

if audio_file is not None:
  # Create a unique timestamp
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  noisy_filename = f"saved_recordings/noisy_{timestamp}.wav"
  cleaned_filename = f"saved_recordings/cleaned_{timestamp}.wav"

  # Read audio data
  data, rate = sf.read(audio_file)

  if len(data.shape) > 1:
    data = np.mean(data, axis=1)

  # Save original noisy audio
  sf.write(noisy_filename, data, rate)

  st.subheader("🔊 Current Recorded Audio:")
  st.audio(noisy_filename, format="audio/wav")

  # Download button for noisy file
  with open(noisy_filename, "rb") as f:
    st.download_button(
        label="📥 Download Noisy Audio",
        data=f,
        file_name=f"noisy_{timestamp}.wav",
        mime="audio/wav",
    )

  if st.button("🧹 Reduce Noise and Save"):
    with st.spinner("⏳ Reducing noise..."):
      # Apply noise reduction
      reduced_audio = nr.reduce_noise(y=data, sr=rate)

      # Save clean audio
      sf.write(cleaned_filename, reduced_audio, rate)

      st.success(
          f"✨ Success! Cleaned file has been saved: `{cleaned_filename}`"
      )

      st.subheader("🎧 Cleaned Audio Output:")
      st.audio(cleaned_filename, format="audio/wav")

      # Download button for cleaned file
      with open(cleaned_filename, "rb") as f:
        st.download_button(
            label="📥 Download Cleaned Audio",
            data=f,
            file_name=f"cleaned_{timestamp}.wav",
            mime="audio/wav",
        )

# --- List of All Saved Recordings ---
st.markdown("---")
st.subheader("📂 All Your Saved Recordings:")

saved_files = sorted(os.listdir("saved_recordings"), reverse=True)

if saved_files:
  for file in saved_files:
    file_path = os.path.join("saved_recordings", file)
    st.text(file)
    st.audio(file_path)

    # Add download button for each file in the history list
    with open(file_path, "rb") as f:
      st.download_button(
          label=f"📥 Download {file}",
          data=f,
          file_name=file,
          mime="audio/wav",
          key=file,  # Unique key for each button in the loop
      )
else:
  st.info("No recordings saved yet.")
