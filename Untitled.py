from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import warnings
import numpy as np
import librosa
import io
import re
warnings.filterwarnings("ignore")

# Initialize the speech recognition pipeline
pipe = pipeline("automatic-speech-recognition", 
                model="nguyenvulebinh/wav2vec2-base-vietnamese-250h", 
                device=0,
                return_timestamps="word")  # Enable word-level timestamps

# Initialize DeepSeek model for text analysis
@st.cache_resource
def load_deepseek_model():
    return pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

st.title("Speech to Text with Content Analysis")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])

# Only process when a file is uploaded
if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file)
    
    # Process the audio file
    with st.spinner("Transcribing audio..."):
        # Convert to bytes and load with librosa
        bytes_data = uploaded_file.getvalue()
        audio_array, sampling_rate = librosa.load(io.BytesIO(bytes_data), sr=16000)
        
        # Process using the pipeline with timestamps
        result = pipe({"raw": audio_array, "sampling_rate": sampling_rate})
        
        # Get transcribed text
        transcribed_text = result["text"]
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', transcribed_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Use DeepSeek for content analysis
        with st.spinner("Analyzing content..."):
            model = load_deepseek_model()
            
            # Analyze each sentence
            inappropriate_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue
                    
                # Prompt for inappropriate content detection at sentence level
                messages = [
                    {"role": "user", "content": f"Is the following sentence inappropriate, offensive, or containing profanity? Answer only with YES or NO.\nSentence: \"{sentence}\""}
                ]
                
                response = model(messages)
                response_text = response[0]["generated_text"]
                
                # Extract the assistant's response
                if "content" in response_text:
                    analysis = response_text.split("content': '")[1].split("'")[0].lower()
                else:
                    analysis = response_text.lower()
                
                # Check if the analysis indicates inappropriate content
                if "yes" in analysis.lower():
                    inappropriate_sentences.append(i)
            
            # Display results
            st.subheader("Transcription Analysis")
            
            if not inappropriate_sentences:
                st.write(transcribed_text)
                st.success("✅ No inappropriate content detected")
            else:
                # Display with highlighted inappropriate sentences
                highlighted_text = ""
                
                for i, sentence in enumerate(sentences):
                    if i in inappropriate_sentences:
                        highlighted_text += f'<span style="background-color:yellow;color:red">{sentence}</span>. '
                    else:
                        highlighted_text += f"{sentence}. "
                
                st.markdown(highlighted_text, unsafe_allow_html=True)
                st.warning(f"⚠️ {len(inappropriate_sentences)} inappropriate sentences detected")
        
        # Display words with timestamps
        st.subheader("Words with timestamps")
        if "chunks" in result:
            words_data = []
            for chunk in result["chunks"]:
                word = chunk["text"]
                start = chunk.get("timestamp", [0, 0])[0]
                end = chunk.get("timestamp", [0, 0])[1]
                words_data.append({"Word": word, "Start (s)": f"{start:.2f}", "End (s)": f"{end:.2f}"})
            
            if words_data:
                st.table(words_data)
            else:
                st.info("No word-level timestamps available")
        else:
            st.info("This model doesn't support word-level timestamps")
else:
    st.info("Please upload an audio file (WAV or MP3) to transcribe")