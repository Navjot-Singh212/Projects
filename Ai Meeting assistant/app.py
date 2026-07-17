import streamlit as st
import os 
import uuid
import whisperx
import warnings
from dotenv import load_dotenv
import gc
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone
from fastembed import TextEmbedding


load_dotenv()
warnings.filterwarnings("ignore")

#configurations
DEVICE="cpu"
COMPUTE_TYPE="int8"
HF_TOKEN=os.getenv("HF_TOKEN")

client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#connecting firebase
if not firebase_admin._apps:
    cred=credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db=firestore.client()

#connecting Pinecone
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.index("meeting-assistant01")

#loading our embedding model
@st.cache_resource
def load_embedding_model():
    em_model=TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return em_model

embedding_model=load_embedding_model()



def format_transcript(rawdata):
    cleaned_segments=""
    for segment in rawdata:
        speaker=segment.get("speaker","UNKNOWN SPEAKER")
        text=segment.get("text","").strip()
        cleaned_segments+=f"{speaker}: {text}\n"
    return cleaned_segments


st.title("AI powered Meeting Assistant")
tab_process, tab_search= st.tabs(["Process and Evaluate new meeting","Search in the database history"])

with tab_process:
    st.markdown("Upload your meeting video/audio to get started~")

    uploaded_file=st.file_uploader("",type=["mp3","wav","m4a","mp4"])

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
    
            del model
            gc.collect()
    

            with st.spinner("Aligning timestamps..."):
                model_a,metadata=whisperx.load_align_model(language_code=result["language"],device=DEVICE)
                result=whisperx.align(result["segments"],model_a,metadata,audio,DEVICE,return_char_alignments=False)
            st.success("Alignment complete!")
            del model_a
            gc.collect()

    
            with st.spinner("Analysing Speakers (Diarization)"):
                diarize_model=whisperx.diarize.DiarizationPipeline(token=HF_TOKEN,device=DEVICE)
                diarize_segments=diarize_model(audio)
                final_result=whisperx.assign_word_speakers(diarize_segments,result)
            st.success("Analysis Complete!")
            del diarize_model
            gc.collect()

            st.subheader("Cleaned Segments:")
            cleaned_captions=format_transcript(final_result["segments"])
            st.text(cleaned_captions)
    
            with st.spinner("Generating Ai Summary and extracting essential keypoints..."):

                prompt=f"""You are an expert executive assistant. Read the following attached transcript and strictly provide only the following mentioned things dont add anything else in the output other than this:
                1. Summary: analyse the meeting and provide a short summary of the meeting.
                2. Key decisions and discussion points: Analyse the meeting transcript and return all the key decisions and dicussion points of the meeting make sure these are in a unordered list(bullet points).
                3. Action items and deadlines: bullet points of action items, tasks assigned and deadlines(if any) mentionin the exact speaker name.
                here is the meeting transcript : {cleaned_captions}
               """
                response=client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                )

                st.divider()
                st.subheader("Ai Meeting Analysis Results:")
                st.markdown(response.text)
            with st.spinner("Saving to Cloud Database..."):
                meeting_id=str(uuid.uuid4())
                meeting_title="Meeting - "+ meeting_id

                doc_ref=db.collection("meetings").document(meeting_id)
                doc_ref.set({
                    "title":meeting_title,
                    "transcript":cleaned_captions,
                    "summary and key points":response.text,
                    "timestamp":firestore.SERVER_TIMESTAMP
                })
                

    