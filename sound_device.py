from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st

st.title("🎙️ Noise Reduction App")
st.write("Record, clean, and manage your audio cleanly.")

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

  if st.button("🧹 Reduce Noise and Save"):
    with st.spinner("⏳ Reducing noise..."):
      # Apply noise reduction
      reduced_audio = nr.reduce_noise(y=data, sr=rate)

      # Save clean audio
      sf.write(cleaned_filename, reduced_audio, rate)

      st.success("✨ Cleaned file saved successfully!")
      st.audio(cleaned_filename, format="audio/wav")

# --- Clean History Section (Using a Dropdown instead of a messy list) ---
st.markdown("---")
st.subheader("📂 Saved Recordings History")

saved_files = sorted(os.listdir("saved_recordings"), reverse=True)

if saved_files:
  # Selectbox keeps the UI clean by showing only one file's options at a time
  selected_file = st.selectbox("Select a recording to view:", saved_files)

  if selected_file:
    file_path = os.path.join("saved_recordings", selected_file)
    st.audio(file_path)

    with open(file_path, "rb") as f:
      st.download_button(
          label=f"📥 Download {selected_file}",
          data=f,
          file_name=selected_file,
          mime="audio/wav",
      )
else:
  st.info("No recordings saved yet.")
