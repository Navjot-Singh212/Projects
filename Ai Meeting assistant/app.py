import streamlit as st
import os 
import json
import whisperx
import warnings
from dotenv import load_dotenv


load_dotenv()
warnings.filterwarnings("ignore")

#configurations
DEVICE="cpu"
COMPUTE_TYPE="int8"
HF_TOKEN=os.getenv("HF_TOKEN")


st.title("AI powered Meeting Assistant~")
st.markdown("Upload your meeting video/audio to get started~")

uploaded_file=st.file_uploader("Upload your video or audio file~",type=["mp3","wav","m4a","mp4"])

if uploaded_file:
    save_path=uploaded_file.name
    
    with open(save_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Successfully saved your file as {save_path}")
    st.audio(save_path)


if st.button("Start AI analysis~"):
    with st.spinner("Transcribing audio...   (this might take a while~)"):
        model=whisperx.load_model("base",DEVICE,compute_type=COMPUTE_TYPE)
        audio=whisperx.load_audio(save_path)
        result=model.transcribe(audio,batch_size=16)
    st.success("Transcription complete!")
    st.info(result)

    with st.spinner("Aligning timestamps..."):
        model_a,metadata=whisperx.load_align_model(language_code=result["language"],device=DEVICE)
        result=whisperx.align(result["segments"],model_a,metadata,audio,DEVICE,return_char_alignments=False)
    st.success("Alignment complete!")
    
    with st.spinner("Analysing Speakers (Diarization)"):
        diarize_model=whisperx.diarize.DiarizationPipeline(token=HF_TOKEN,device=DEVICE)
        diarize_segments=diarize_model(audio)
        final_result=whisperx.assign_word_speakers(diarize_segments,result)
    st.success("Analysis Complete!")

    st.subheader("Raw output data:")
    st.json(final_result["segments"])

