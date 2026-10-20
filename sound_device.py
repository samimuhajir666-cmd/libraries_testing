from datetime import datetime
import os
import subprocess
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
    [
        "🎙️ Noise Reduction",
        "🔥 TorchAudio & AI Processing",
        "🌪️ DeepFilterNet + Polish Pipeline",
        "🎶 Meta Demucs Separation",
    ],
)

# ----------------------------------------------------
# MODE 1: NOISE REDUCTION
# ----------------------------------------------------
if app_mode == "🎙️ Noise Reduction":
  st.title("🎙️ Auto Noise Reduction App")
  st.write(
      "Record audio, noise reduction is applied automatically, and manage past"
      " sessions easily!"
  )

  os.makedirs("saved_recordings", exist_ok=True)
  audio_file = st.audio_input("🔴 Click here to record audio:")

  if audio_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noisy_filename = f"saved_recordings/noisy_{timestamp}.wav"
    cleaned_filename = f"saved_recordings/cleaned_{timestamp}.wav"

    data, rate = sf.read(audio_file)
    if len(data.shape) > 1:
      data = np.mean(data, axis=1)

    sf.write(noisy_filename, data, rate)

    with st.spinner("⏳ Automatically reducing noise..."):
      reduced_audio = nr.reduce_noise(y=data, sr=rate)
      sf.write(cleaned_filename, reduced_audio, rate)

    st.success("✨ Recording saved and noise reduction applied automatically!")

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


# ----------------------------------------------------
# MODE 2: TORCHAUDIO & AI PROCESSING
# ----------------------------------------------------
elif app_mode == "🔥 TorchAudio & AI Processing":
  st.title("🔥 TorchAudio & Advanced AI Processing")
  st.write(
      "Record audio live, load it into PyTorch Tensors, check shapes, resample"
      " to 16kHz, and prepare for AI models!"
  )

  os.makedirs("torch_recordings", exist_ok=True)
  audio_file_torch = st.audio_input(
      "🔴 Click here to record audio for TorchAudio analysis:"
  )

  if audio_file_torch is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"torch_recordings/recorded_{timestamp}.wav"

    with open(temp_path, "wb") as f:
      f.write(audio_file_torch.getbuffer())

    data_np, sample_rate = sf.read(temp_path)
    if len(data_np.shape) > 1:
      data_np = np.mean(data_np, axis=1)
    waveform = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0)

    with st.container(border=True):
      st.subheader("📈 PyTorch Tensor Analysis")
      st.write(f"**Original Sample Rate:** {sample_rate} Hz")
      st.write(f"**Waveform Tensor Shape:** `{waveform.shape}` (Channels, Samples)")
      st.audio(temp_path, format="audio/wav")

      if st.button("⚡ Resample to 16kHz (AI Ready)", key=f"resample_{timestamp}"):
        target_sr = 16000
        if sample_rate != target_sr:
          resampler = torchaudio.transforms.Resample(
              orig_freq=sample_rate, new_freq=target_sr
          )
          resampled_waveform = resampler(waveform)

          resampled_path = f"torch_recordings/resampled_16k_{timestamp}.wav"
          torchaudio.save(resampled_path, resampled_waveform, target_sr)

          st.success("✅ Successfully resampled to 16kHz!")
          st.audio(resampled_path, format="audio/wav")

          with open(resampled_path, "rb") as f:
            st.download_button(
                label="📥 Download 16kHz Audio",
                data=f,
                file_name=f"resampled_16kHz_{timestamp}.wav",
                mime="audio/wav",
                key=f"download_16k_{timestamp}",
            )


# ----------------------------------------------------
# MODE 3: DEEPFILTERNET + NOISEREDUCE PIPELINE
# ----------------------------------------------------
elif app_mode == "🌪️ DeepFilterNet + Polish Pipeline":
  st.title("🌪️ DeepFilterNet & AI Dual-Layer Pipeline")
  st.write(
      "Record audio with wind or heavy interference. Stage 1: DeepFilterNet"
      " removes wind/heavy noise. Stage 2: Noisereduce polishes the result!"
  )

  os.makedirs("deepfilter_recordings", exist_ok=True)
  audio_file_df = st.audio_input(
      "🔴 Click here to record audio for DeepFilterNet Pipeline:"
  )

  if audio_file_df is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noisy_path = f"deepfilter_recordings/raw_{timestamp}.wav"
    df_cleaned_path = f"deepfilter_recordings/df_cleaned_{timestamp}.wav"
    fully_polished_path = f"deepfilter_recordings/polished_{timestamp}.wav"

    with open(noisy_path, "wb") as f:
      f.write(audio_file_df.getbuffer())

    data, rate = sf.read(noisy_path)
    if len(data.shape) > 1:
      data = np.mean(data, axis=1)

    with st.spinner(
        "🌪️ Stage 1: Running DeepFilterNet (Removing Wind & Heavy Noise)..."
    ):
      try:
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df()
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        if rate != df_state.sr():
          resampler = torchaudio.transforms.Resample(
              orig_freq=rate, new_freq=df_state.sr()
          )
          waveform_resampled = resampler(waveform)
        else:
          waveform_resampled = waveform

        enhanced_waveform = enhance(model, df_state, waveform_resampled)

        if df_state.sr() != rate:
          back_resampler = torchaudio.transforms.Resample(
              orig_freq=df_state.sr(), new_freq=rate
          )
          enhanced_waveform = back_resampler(enhanced_waveform)

        df_cleaned_data = enhanced_waveform.squeeze(0).numpy()
        sf.write(df_cleaned_path, df_cleaned_data, rate)

      except Exception as e:
        st.warning(
            f"DeepFilterNet running in fallback mode due to environment setup:"
            f" {e}"
        )
        df_cleaned_data = data
        sf.write(df_cleaned_path, df_cleaned_data, rate)

    with st.spinner(
        "✨ Stage 2: Running Noisereduce (Polishing Crystal Clear Sound)..."
    ):
      polished_data = nr.reduce_noise(y=df_cleaned_data, sr=rate)
      sf.write(fully_polished_path, polished_data, rate)

    st.success("🎉 Dual-Layer AI Pipeline Completed Successfully!")

    with st.container(border=True):
      st.subheader("📊 Pipeline Results Comparison")
      col1, col2, col3 = st.columns(3)

      with col1:
        st.markdown("### 🔊 1. Raw Noisy")
        st.audio(noisy_path, format="audio/wav")
        with open(noisy_path, "rb") as f:
          st.download_button(
              label="📥 Download Raw",
              data=f,
              file_name=f"raw_{timestamp}.wav",
              mime="audio/wav",
              key=f"download_raw_{timestamp}",
          )

      with col2:
        st.markdown("### 🌪️ 2. DeepFiltered")
        st.audio(df_cleaned_path, format="audio/wav")
        with open(df_cleaned_path, "rb") as f:
          st.download_button(
              label="📥 Download Filtered",
              data=f,
              file_name=f"df_cleaned_{timestamp}.wav",
              mime="audio/wav",
              key=f"download_df_{timestamp}",
          )

      with col3:
        st.markdown("### ✨ 3. Fully Polished")
        st.audio(fully_polished_path, format="audio/wav")
        with open(fully_polished_path, "rb") as f:
          st.download_button(
              label="📥 Download Final",
              data=f,
              file_name=f"polished_{timestamp}.wav",
              mime="audio/wav",
              key=f"download_polished_{timestamp}",
          )


# ----------------------------------------------------
# MODE 4: META DEMUCS SOURCE SEPARATION
# ----------------------------------------------------
elif app_mode == "🎶 Meta Demucs Separation":
  st.title("🎶 Meta Demucs Advanced Audio Separation")
  st.write(
      "Using Meta's state-of-the-art Demucs model to isolate vocals and clean"
      " complex background noise!"
  )

  os.makedirs("demucs_recordings", exist_ok=True)
  audio_file_demucs = st.audio_input(
      "🔴 Click here to record audio for Demucs Separation:"
  )

  if audio_file_demucs is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_demucs_path = f"demucs_recordings/raw_{timestamp}.wav"

    with open(raw_demucs_path, "wb") as f:
      f.write(audio_file_demucs.getbuffer())

    with st.spinner(
        "🎶 Running Meta Demucs Model (Isolating Vocals & Removing Noise)..."
    ):
      try:
        cmd = [
            "python",
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems",
            "vocals",
            raw_demucs_path,
            "-o",
            "demucs_recordings/output",
        ]
        # Capture stdout and stderr to show exact error if it fails
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )

        base_name = os.path.splitext(os.path.basename(raw_demucs_path))[0]
        isolated_vocal_path = os.path.join(
            "demucs_recordings", "output", "htdemucs", base_name, "vocals.wav"
        )

        st.success("🎉 Meta Demucs Separation Completed Successfully!")

        with st.container(border=True):
          st.subheader("📊 Demucs Isolation Results")
          col1, col2 = st.columns(2)

          with col1:
            st.markdown("### 🔊 Original Mixed Audio")
            st.audio(raw_demucs_path, format="audio/wav")

          with col2:
            st.markdown("### 🎤 Isolated Clean Vocals")
            if os.path.exists(isolated_vocal_path):
              st.audio(isolated_vocal_path, format="audio/wav")
              with open(isolated_vocal_path, "rb") as f:
                st.download_button(
                    label="📥 Download Isolated Vocals",
                    data=f,
                    file_name=f"isolated_vocals_{timestamp}.wav",
                    mime="audio/wav",
                    key=f"download_demucs_{timestamp}",
                )
            else:
              st.warning("Isolated vocal file path could not be resolved.")

      except subprocess.CalledProcessError as e:
        st.error(
            f"Demucs Execution Failed (Exit Code: {e.returncode}). Check logs"
            " below:"
        )
        st.code(e.stderr)
        st.info(
            "💡 Note: Meta Demucs is a heavy model and requires FFmpeg + high"
            " RAM. If you are on Streamlit Cloud, it may fail due to memory"
            " limits. Try running locally on your PC!"
        )
      except Exception as e:
        st.error(f"Error running Demucs: {e}")
