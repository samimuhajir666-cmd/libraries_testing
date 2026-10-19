from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

# Page configuration
st.set_page_config(
    page_title="AI Audio Studio", page_icon="🎙️", layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎛️ Navigation Menu")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    ["🎙️ Noise Reduction", "🔥 TorchAudio & AI Processing"],
)

# ----------------------------------------------------
# MODE 1: NOISE REDUCTION (Your Existing Code)
# ----------------------------------------------------
if app_mode == "🎙️ Noise Reduction":
  st.title("🎙️ Auto Noise Reduction App")
  st.write(
      "Record audio, noise reduction is applied automatically, and manage past"
      " sessions easily!"
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

    # Display latest audio players side-by-side using a bordered container
    with st.container(border=True):
      st.subheader("📊 Latest Session Output")
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

  if os.path.exists("saved_recordings"):
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
          "Choose a session to view or download:",
          timestamps,
          format_func=lambda x: f"Session: {x[:4]}-{x[4:6]}-{x[6:8]} at {x[9:11]}:{x[11:13]}:{x[13:15]}",
          key="past_session_select",
      )

      if selected_session:
        noisy_path = os.path.join(
            "saved_recordings", f"noisy_{selected_session}.wav"
        )
        cleaned_path = os.path.join(
            "saved_recordings", f"cleaned_{selected_session}.wav"
        )

        with st.container(border=True):
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
            else:
              st.warning("Noisy file not found.")

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
              st.warning("Cleaned file not found.")
    else:
      st.info("No past sessions found yet.")
  else:
    st.info("No recordings directory found.")


# ----------------------------------------------------
# MODE 2: TORCHAUDIO & AI PROCESSING (New Section)
# ----------------------------------------------------
elif app_mode == "🔥 TorchAudio & AI Processing":
  st.title("🔥 TorchAudio & Advanced AI Processing")
  st.write(
      "Load audio into PyTorch Tensors, check shapes, resample to 16kHz, and"
      " prepare for AI models!"
  )

  # Create folder for torch outputs
  os.makedirs("torch_recordings", exist_ok=True)

  uploaded_torch_file = st.file_uploader(
      "Upload an audio file for TorchAudio analysis:", type=["wav", "mp3"]
  )

  if uploaded_torch_file is not None:
    # Save temporarily to process with torchaudio
    temp_path = "torch_recordings/temp_uploaded.wav"
    with open(temp_path, "wb") as f:
      f.write(uploaded_torch_file.getbuffer())

    # Load with Torchaudio
    waveform, sample_rate = torchaudio.load(temp_path)

    with st.container(border=True):
      st.subheader("📈 PyTorch Tensor Analysis")
      st.write(f"**Original Sample Rate:** {sample_rate} Hz")
      st.write(f"**Waveform Tensor Shape:** `{waveform.shape}` (Channels, Samples)")

      # Play audio
      st.audio(temp_path, format="audio/wav")

      # Resampling option to 16kHz for AI models
      if st.button("⚡ Resample to 16kHz (AI Ready)"):
        target_sr = 16000
        if sample_rate != target_sr:
          resampler = torchaudio.transforms.Resample(
              orig_freq=sample_rate, new_freq=target_sr
          )
          resampled_waveform = resampler(waveform)

          resampled_path = "torch_recordings/resampled_16k.wav"
          torchaudio.save(resampled_path, resampled_waveform, target_sr)

          st.success("✅ Successfully resampled to 16kHz!")
          st.write(
              f"**New Waveform Shape:** `{resampled_waveform.shape}` at"
              f" {target_sr} Hz"
          )
          st.audio(resampled_path, format="audio/wav")

          with open(resampled_path, "rb") as f:
            st.download_button(
                label="📥 Download 16kHz Audio",
                data=f,
                file_name="resampled_16kHz.wav",
                mime="audio/wav",
            )
        else:
          st.info("Audio is already at 16kHz sample rate!")from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

# Page configuration
st.set_page_config(
    page_title="AI Audio Studio", page_icon="🎙️", layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎛️ Navigation Menu")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    ["🎙️ Noise Reduction", "🔥 TorchAudio & AI Processing"],
)

# ----------------------------------------------------
# MODE 1: NOISE REDUCTION (Your Existing Code)
# ----------------------------------------------------
if app_mode == "🎙️ Noise Reduction":
  st.title("🎙️ Auto Noise Reduction App")
  st.write(
      "Record audio, noise reduction is applied automatically, and manage past"
      " sessions easily!"
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

    # Display latest audio players side-by-side using a bordered container
    with st.container(border=True):
      st.subheader("📊 Latest Session Output")
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

  if os.path.exists("saved_recordings"):
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
          "Choose a session to view or download:",
          timestamps,
          format_func=lambda x: f"Session: {x[:4]}-{x[4:6]}-{x[6:8]} at {x[9:11]}:{x[11:13]}:{x[13:15]}",
          key="past_session_select",
      )

      if selected_session:
        noisy_path = os.path.join(
            "saved_recordings", f"noisy_{selected_session}.wav"
        )
        cleaned_path = os.path.join(
            "saved_recordings", f"cleaned_{selected_session}.wav"
        )

        with st.container(border=True):
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
            else:
              st.warning("Noisy file not found.")

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
              st.warning("Cleaned file not found.")
    else:
      st.info("No past sessions found yet.")
  else:
    st.info("No recordings directory found.")


# ----------------------------------------------------
# MODE 2: TORCHAUDIO & AI PROCESSING (New Section)
# ----------------------------------------------------
elif app_mode == "🔥 TorchAudio & AI Processing":
  st.title("🔥 TorchAudio & Advanced AI Processing")
  st.write(
      "Load audio into PyTorch Tensors, check shapes, resample to 16kHz, and"
      " prepare for AI models!"
  )

  # Create folder for torch outputs
  os.makedirs("torch_recordings", exist_ok=True)

  uploaded_torch_file = st.file_uploader(
      "Upload an audio file for TorchAudio analysis:", type=["wav", "mp3"]
  )

  if uploaded_torch_file is not None:
    # Save temporarily to process with torchaudio
    temp_path = "torch_recordings/temp_uploaded.wav"
    with open(temp_path, "wb") as f:
      f.write(uploaded_torch_file.getbuffer())

    # Load with Torchaudio
    waveform, sample_rate = torchaudio.load(temp_path)

    with st.container(border=True):
      st.subheader("📈 PyTorch Tensor Analysis")
      st.write(f"**Original Sample Rate:** {sample_rate} Hz")
      st.write(f"**Waveform Tensor Shape:** `{waveform.shape}` (Channels, Samples)")

      # Play audio
      st.audio(temp_path, format="audio/wav")

      # Resampling option to 16kHz for AI models
      if st.button("⚡ Resample to 16kHz (AI Ready)"):
        target_sr = 16000
        if sample_rate != target_sr:
          resampler = torchaudio.transforms.Resample(
              orig_freq=sample_rate, new_freq=target_sr
          )
          resampled_waveform = resampler(waveform)

          resampled_path = "torch_recordings/resampled_16k.wav"
          torchaudio.save(resampled_path, resampled_waveform, target_sr)

          st.success("✅ Successfully resampled to 16kHz!")
          st.write(
              f"**New Waveform Shape:** `{resampled_waveform.shape}` at"
              f" {target_sr} Hz"
          )
          st.audio(resampled_path, format="audio/wav")

          with open(resampled_path, "rb") as f:
            st.download_button(
                label="📥 Download 16kHz Audio",
                data=f,
                file_name="resampled_16kHz.wav",
                mime="audio/wav",
            )
        else:
          st.info("Audio is already at 16kHz sample rate!")from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

# Page configuration
st.set_page_config(
    page_title="AI Audio Studio", page_icon="🎙️", layout="wide"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎛️ Navigation Menu")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    ["🎙️ Noise Reduction", "🔥 TorchAudio & AI Processing"],
)

# ----------------------------------------------------
# MODE 1: NOISE REDUCTION (Your Existing Code)
# ----------------------------------------------------
if app_mode == "🎙️ Noise Reduction":
  st.title("🎙️ Auto Noise Reduction App")
  st.write(
      "Record audio, noise reduction is applied automatically, and manage past"
      " sessions easily!"
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

    # Display latest audio players side-by-side using a bordered container
    with st.container(border=True):
      st.subheader("📊 Latest Session Output")
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

  if os.path.exists("saved_recordings"):
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
          "Choose a session to view or download:",
          timestamps,
          format_func=lambda x: f"Session: {x[:4]}-{x[4:6]}-{x[6:8]} at {x[9:11]}:{x[11:13]}:{x[13:15]}",
          key="past_session_select",
      )

      if selected_session:
        noisy_path = os.path.join(
            "saved_recordings", f"noisy_{selected_session}.wav"
        )
        cleaned_path = os.path.join(
            "saved_recordings", f"cleaned_{selected_session}.wav"
        )

        with st.container(border=True):
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
            else:
              st.warning("Noisy file not found.")

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
              st.warning("Cleaned file not found.")
    else:
      st.info("No past sessions found yet.")
  else:
    st.info("No recordings directory found.")


# ----------------------------------------------------
# MODE 2: TORCHAUDIO & AI PROCESSING (New Section)
# ----------------------------------------------------
elif app_mode == "🔥 TorchAudio & AI Processing":
  st.title("🔥 TorchAudio & Advanced AI Processing")
  st.write(
      "Load audio into PyTorch Tensors, check shapes, resample to 16kHz, and"
      " prepare for AI models!"
  )

  # Create folder for torch outputs
  os.makedirs("torch_recordings", exist_ok=True)

  uploaded_torch_file = st.file_uploader(
      "Upload an audio file for TorchAudio analysis:", type=["wav", "mp3"]
  )

  if uploaded_torch_file is not None:
    # Save temporarily to process with torchaudio
    temp_path = "torch_recordings/temp_uploaded.wav"
    with open(temp_path, "wb") as f:
      f.write(uploaded_torch_file.getbuffer())

    # Load with Torchaudio
    waveform, sample_rate = torchaudio.load(temp_path)

    with st.container(border=True):
      st.subheader("📈 PyTorch Tensor Analysis")
      st.write(f"**Original Sample Rate:** {sample_rate} Hz")
      st.write(f"**Waveform Tensor Shape:** `{waveform.shape}` (Channels, Samples)")

      # Play audio
      st.audio(temp_path, format="audio/wav")

      # Resampling option to 16kHz for AI models
      if st.button("⚡ Resample to 16kHz (AI Ready)"):
        target_sr = 16000
        if sample_rate != target_sr:
          resampler = torchaudio.transforms.Resample(
              orig_freq=sample_rate, new_freq=target_sr
          )
          resampled_waveform = resampler(waveform)

          resampled_path = "torch_recordings/resampled_16k.wav"
          torchaudio.save(resampled_path, resampled_waveform, target_sr)

          st.success("✅ Successfully resampled to 16kHz!")
          st.write(
              f"**New Waveform Shape:** `{resampled_waveform.shape}` at"
              f" {target_sr} Hz"
          )
          st.audio(resampled_path, format="audio/wav")

          with open(resampled_path, "rb") as f:
            st.download_button(
                label="📥 Download 16kHz Audio",
                data=f,
                file_name="resampled_16kHz.wav",
                mime="audio/wav",
            )
        else:
          st.info("Audio is already at 16kHz sample rate!")
