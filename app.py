import streamlit as st
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
import os

from model import GPTLanguageModel, generate

# Disable Streamlit's file watcher for torch modules to prevent RuntimeError
if 'STREAMLIT_WATCH_MODULES' not in os.environ:
    os.environ['STREAMLIT_WATCH_MODULES'] = 'false'


# Define model parameters

num_train_epochs = 50
batch_size = 512
step_token = 1
context_len = 50
embedding_dim = 100
head_size=8
num_heads = 8
num_transformer_blocks = 5
dropout = 0.2
train_split_percent = 0.75
generate_temperature = 0.8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Set page title and description
st.title("Persian Poetry Generator")
st.markdown("Generate Persian poetry in the style of Hafez using a transformer model")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = Tokenizer.from_file("persian_poetry_bpe.json")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")
    
    # Load saved model
    try:
        vocab_size = tokenizer.get_vocab_size()
        
        # Create model instance
        model = GPTLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim, context_len=context_len, num_transformer_blocks=num_transformer_blocks, head_size=head_size, num_heads=num_heads, dropout=dropout)
        
        # Load model weights (state dict) into the model
        state_dict = torch.load("model_checkpoint_final.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, context_len
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Check if your model structure matches the expected format.")
        raise e


# Load model and tokenizer
try:
    # Display loading message
    with st.spinner("Loading model and tokenizer..."):
        model, tokenizer, device, context_len = load_model_and_tokenizer()
    
    st.success("Model loaded successfully!")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Troubleshooting tips:")
    st.markdown("""
    1. Make sure `model_checkpoint_final.pt` and `persian_poetry_bpe.json` are in the same directory as this app.
    2. Check if your model was saved with a compatible PyTorch version.
    3. Try running with `--server.enableCORS=false` flag.
    4. Run Streamlit with the debug flag: `streamlit run --logger.level=debug app.py`
    """)
    model_loaded = False

# User interface
if model_loaded:
    # Input for prompt
    prompt = st.text_area("Starting prompt:", 
                          value="از چشم بخت خویش مبادت گزند", 
                          help="Enter the starting text for poetry generation")
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Maximum new tokens:", min_value=10, max_value=context_len, value=50)
    with col2:
        temperature = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
                               help="Higher values produce more random outputs, lower values more deterministic ones")
    
    # Generate button
    if st.button("Generate Poetry"):
        with st.spinner("Generating poetry..."):
            # Encode the prompt
            prompt_ids = tokenizer.encode(prompt).ids
            
            print(prompt_ids)
            # Generate poem continuation
            generated = generate(model, tokenizer, prompt_ids, max_new_tokens=max_tokens, 
                                context_len=context_len, temperature=temperature, device=device)
            
            # Decode the full generated text
            full_poem = tokenizer.decode(generated[0].tolist())
            
            # Display result
            st.subheader("Generated Poetry:")
            st.write(full_poem)
            
            # Show token IDs if requested
            if st.checkbox("Show token IDs"):
                st.json(generated[0].tolist())
else:
    st.warning("Please make sure model and tokenizer files are available in the same directory as this app.")

# How to run instructions
st.markdown("---")
st.markdown("### How to run this app:")
st.code("streamlit run app.py", language="bash") 