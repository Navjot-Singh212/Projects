import streamlit as st
import os 
import uuid
import warnings
from dotenv import load_dotenv
import gc
# import whisperx
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone
from fastembed import TextEmbedding
import streamlit_authenticator as stauth


load_dotenv()
warnings.filterwarnings("ignore")

# #configurations
# DEVICE="cpu"
# COMPUTE_TYPE="int8"
# HF_TOKEN=st.secrets["HF_TOKEN"]

client=genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

#connecting firebase
if not firebase_admin._apps:
    firebase_cred=dict(st.secrets["firebase"])
    cred=credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred)
db=firestore.client()

def transcribe_audio(audio_file_path):
    uploaded_audio=client.files.upload(file=audio_file_path)
    prompt = """
    Process this meeting audio file and generate a detailed transcription.
    Requirements:
    1. Identify distinct speakers (e.g., Speaker 1, Speaker 2).
    2. Provide timestamps for the conversation segments.
    3. Output the transcript in a clean, readable format.
    """
    response=client.models.generate_content(model="gemini-3.1-flash-lite",contents=[prompt,uploaded_audio])
    return response.text


#connecting Pinecone
pc=Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index=pc.Index("meeting-assistant01")

#loading our embedding model
@st.cache_resource
def load_embedding_model():
    em_model=TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return em_model

embedding_model=load_embedding_model()



# def format_transcript(rawdata):
#     cleaned_segments=""
#     for segment in rawdata:
#         speaker=segment.get("speaker","UNKNOWN SPEAKER")
#         text=segment.get("text","").strip()
#         cleaned_segments+=f"{speaker}: {text}\n"
#     return cleaned_segments


def chunk_transcript(text,words_per_chunk=100):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),words_per_chunk):
        chunk=" ".join(words[i:i+words_per_chunk])
        chunks.append(chunk)
    return chunks

#Firestore authentication
credential_dict={"usernames": {}}

user_ref=db.collection("users")
docs=user_ref.stream()

for doc in docs:
    user_id=doc.id
    user_data=doc.to_dict()
    credential_dict["usernames"][user_id]=user_data

cookie_config={
    "cookie":{
        "name":"meeting assistant cookie",
        "expiry_days":30,
        "key":st.secrets["COOKIE_KEY"]
    },
    "preauthorised":{
        "emails":[]
    }
}
authenticator=stauth.Authenticate(credentials=credential_dict,
                                  cookie_expiry_days=cookie_config["cookie"]["expiry_days"],
                                  cookie_key=cookie_config["cookie"]["key"],
                                  cookie_name=cookie_config["cookie"]["name"],)

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"]=None

if not st.session_state["authentication_status"]:
    auth_mode=st.sidebar.pills("Welcome!",["Login","Register a new Account"],default="Login",label_visibility="hidden")

    if auth_mode=="Register a new Account":
        try:
            email_reg, username_reg, name_reg=authenticator.register_user()

            if email_reg:
                st.toast("User registered successfully!! \n(Please move to login~) ")

                new_user_entry=credential_dict["usernames"][username_reg]
                db.collection("users").document(username_reg).set(new_user_entry)

        except Exception as e:
            st.sidebar.error(e)
    elif auth_mode=="Login":
        authenticator.login()


if st.session_state["authentication_status"] is False:
    st.error("Username/password incorrect")
if st.session_state["authentication_status"] is None:
    st.warning("Please enter all the login details~")

if st.session_state["authentication_status"]:
    username=st.session_state["username"]
    user_namespace=f"user_{st.session_state['username']}"
    st.sidebar.title(f"Welcome, {st.session_state['name']}")
    authenticator.logout("Logout",location="sidebar")
    st.sidebar.info(f"Your data is securely stored in {user_namespace}")



    st.title("AI powered Meeting Assistant")
    tab_process, tab_search,tab_dashboard,tab_chat= st.tabs(["Process and Evaluate new meeting","Search in the database history","Meeting Dashboard","Ai Chat"])

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
                with st.spinner("Processing audio using Gemini AI..."):
                    cleaned_captions=transcribe_audio(save_path)

                "this is the code for transcribing audio, aligning timestamps, and Diarization but i couldn't use this as the whisperx library size was too big for deploying on any free service..."
                # with st.spinner("Transcribing audio...   (this might take a while~)"):
                #     model=whisperx.load_model("base",DEVICE,compute_type=COMPUTE_TYPE)
                #     audio=whisperx.load_audio(save_path)
                #     result=model.transcribe(audio,batch_size=16)
                # st.success("Transcription complete!")
        
                # del model
                # gc.collect()
        

                # with st.spinner("Aligning timestamps..."):
                #     model_a,metadata=whisperx.load_align_model(language_code=result["language"],device=DEVICE)
                #     result=whisperx.align(result["segments"],model_a,metadata,audio,DEVICE,return_char_alignments=False)
                # st.success("Alignment complete!")
                # del model_a
                # gc.collect()

        
                # with st.spinner("Analysing Speakers (Diarization)"):
                #     diarize_model=whisperx.diarize.DiarizationPipeline(token=HF_TOKEN,device=DEVICE)
                #     diarize_segments=diarize_model(audio)
                #     final_result=whisperx.assign_word_speakers(diarize_segments,result)
                st.success("Analysis Complete!")
                # del diarize_model
                # gc.collect()

                st.subheader("Cleaned Segments:")
                # cleaned_captions=format_transcript(final_result["segments"])
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
                        "user_id":username,
                        "title":meeting_title,
                        "transcript":cleaned_captions,
                        "summary and key points":response.text,
                        "timestamp":firestore.SERVER_TIMESTAMP
                    })
                    
                    searchable_data=f"Summary: {response.text}\n\n Transcript: {cleaned_captions}"
                    chunks=chunk_transcript(searchable_data)
                    embeddings_generator=embedding_model.embed(chunks)
                    meeting_vector=list(embeddings_generator)

                    pinecone_to_upsert=[]
                    for  idx,vector in enumerate(meeting_vector):
                        pinecone_to_upsert.append({
                            "id":f"{meeting_id}_chunk_{idx}",
                            "values":vector.tolist(),
                            "metadata":{
                                "meeting_id":meeting_id,
                                "title":meeting_title,
                                "content":chunks[idx]
                            }

                        })
                    index.upsert(vectors=pinecone_to_upsert,namespace=user_namespace)
                    st.success("Meeting securely stored on the cloud~")


    with tab_search:
        st.header("Search Past Meetings")
        search_query=st.text_input("Ask a question or search a topic...(eg: What was the budget for the 'tasty pasta' project)")    

        if "search_results" not in st.session_state:
            st.session_state.search_results=None



        if st.button("Search meetings~"):
            if search_query:
                with st.spinner("Searching..."):
                    query_generator=embedding_model.embed([search_query])
                    query_vector=list(query_generator)[0].tolist()

                    st.session_state.search_results=index.query(vector=query_vector,include_metadata=True,top_k=3,namespace=user_namespace)
                
                                    
                                    

            else:
                st.warning("Please enter a search term first!!!")

        if st.session_state.search_results:
            st.success("Search Complete!")
                
            for match in st.session_state.search_results["matches"]:
                metadata=match["metadata"]
                similarity_score=match["score"]
                chunk_id=match["id"]

                matched_snippet=metadata.get("content","No text snippet attached")
                matched_title=metadata.get("title","Unkown Title")
                matched_meeting_id=metadata.get("meeting_id")
                    
                with st.expander(f"{matched_title}    Match score: {similarity_score:.2f}"):
                    st.markdown("**Exact Meeting segment:**")
                    st.info(f"'{matched_snippet}'")
                    st.caption(f"Source Meeting id: {matched_meeting_id}")

                    if st.button("Load Full Meeting Record",key=f"fetch_{chunk_id}"):
                        with st.spinner("Retrieving full document from the Cloud..."):
                            doc_ref=db.collection("meetings").document(matched_meeting_id)
                            doc=doc_ref.get()

                            if doc.exists:
                                complete_data=doc.to_dict()
                                if complete_data.get("user_id")==username:
                                    st.divider()
                                    st.subheader(complete_data.get("title"),"")
                                    st.markdown("### AI Meeting Summary")
                                    st.markdown(complete_data.get("summary and key points","No summary found!"))
                                    st.markdown("Full Transcript")
                                    st.text_area("Transcript:",value=complete_data.get("transcript","No transcript found!"))

                                else:
                                    st.error("ACCESS DENIED: You do not have permission to view this meeting file!!!")
                                    
                            else:
                                st.error("Could not find the master document in the Cloud!")
                                
    with tab_dashboard:
        st.header("Meeting Dasboard")

        user_doc_ref=db.collection("meetings").where("user_id","==",username).stream()
        meeting_list=list(user_doc_ref)

        if len(meeting_list)==0:
            st.info("You havent uploaded any meetings yet!\nAdd some to browse through your meeting's data here~")
        else:
            st.success(f"Found {len(meeting_list)} meetings in your secret inventory!")

            for doc in meeting_list:
                data=doc.to_dict()
                title=data.get("title","Unkown title")
                
                with st.expander(title):
                    st.markdown("AI Summary:")
                    st.markdown(data.get("summary and key points","No Summary Found!"))
                    st.markdown("Full Meeting Transcript:")
                    st.text_area("Raw Text:",value=data.get("transcript","No Transcript Found!"),height=150,key=f"dashehe_{doc.id}")
                
    with tab_chat:
        st.header("Chat about your meetings")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages=[]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt:=st.chat_input("Ask a question about any of your past meetings~"):

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_messages.append({"role":"user","content":prompt})

            with st.chat_message("assistant"):
                with st.spinner("The Ai is fetching the data from the cloud and is thinking...."):
                    user_query=list(embedding_model.embed([prompt]))[0].tolist()
                    search_results=index.query(top_k=3,vector=user_query,namespace=user_namespace,include_metadata=True)

                    retrived_data=""
                    for match in search_results["matches"]:
                        title=match["metadata"].get("title","Unknown Title")
                        match_content=match["metadata"].get("content","")
                        retrived_data+=f"From {title}:\n{match_content}\n\n"
                    
                    ai_prompt = f"""You are an AI executive assistant. Answer the user's question using ONLY the context provided below from their past meetings. 
                    If the context doesn't contain the answer, politely say "I cannot find that in your meeting history."

                    CONTEXT FROM PAST MEETINGS:
                    {retrived_data}
                
                    USER QUESTION:
                    {prompt}
                    """
                    response=client.models.generate_content(model="gemini-3.1-flash-lite",contents=ai_prompt)
                    st.markdown(response.text)
                    st.session_state.chat_messages.append({"role":"assistant","content":response.text}) 





    