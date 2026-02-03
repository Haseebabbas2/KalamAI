import os
import torch
import numpy as np
import torch.nn as nn
import streamlit as st
from PIL import Image

# Define the CharRNN model class
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout_rate=0.3):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout_rate = dropout_rate

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = output.contiguous().view(-1, output.shape[2])
        logits = self.fc(output)
        return logits, hidden

# Loading the model
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = CharRNN(checkpoint['vocab_size'],
                            checkpoint['embed_size'],
                            checkpoint['hidden_size'],
                            checkpoint['num_layers'],
                            dropout_rate=checkpoint['dropout_rate']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    return loaded_model, char2idx, idx2char

# Define NEWLINE_TOKEN
NEWLINE_TOKEN = "<NEWLINE>"

# Text Generation
def generate_text(model, start_text, char2idx, idx2char, generation_length=200, temperature=0.8):
    model.eval()
    input_indices = [char2idx.get(ch, 0) for ch in start_text]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    hidden = None
    generated_text = start_text

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_tensor, hidden)
            logits = logits[-1] / temperature
            probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)

            next_char = idx2char[next_char_idx]
            generated_text += next_char

            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    # Replace <NEWLINE> token with actual newline
    generated_text = generated_text.replace(NEWLINE_TOKEN, '\n')

    return generated_text


# Load the *best* model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'my_char_rnn_model.pth'  # Ensure this is the correct path to your model file
loaded_model, loaded_char2idx, loaded_idx2char = load_model(model_path, device)

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI - Modern Dark Theme with Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════════

# Page Configuration
st.set_page_config(
    page_title="KalamAI | AI Poetry Generator",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Noto+Nastaliq+Urdu:wght@400;500;600;700&display=swap');
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Title */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f5af19 0%, #f12711 50%, #c471ed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(241, 39, 17, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(245, 175, 25, 0.3)); }
        to { filter: drop-shadow(0 0 25px rgba(196, 113, 237, 0.5)); }
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .hero-icon {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f5af19 !important;
        box-shadow: 0 0 20px rgba(245, 175, 25, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #f5af19, #f12711, #c471ed) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 2px solid #f5af19 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #f5af19 0%, #f12711 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(241, 39, 17, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(241, 39, 17, 0.4) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: #f5af19 !important;
    }
    
    /* Poetry output container */
    .poetry-output {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Noto Nastaliq Urdu', 'Inter', serif;
        font-size: 1.2rem;
        line-height: 2.2;
        color: rgba(255, 255, 255, 0.9);
        white-space: pre-wrap;
        text-align: right;
        direction: rtl;
        min-height: 200px;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .poetry-output::before {
        content: '❝';
        font-size: 3rem;
        color: rgba(245, 175, 25, 0.3);
        position: absolute;
        top: -10px;
        right: 10px;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(48, 43, 99, 0.95) 0%, rgba(15, 12, 41, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    .sidebar-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.7;
    }
    
    .tip-card {
        background: rgba(245, 175, 25, 0.1);
        border-left: 3px solid #f5af19;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    /* Labels */
    .stTextInput label, .stSlider label {
        font-family: 'Inter', sans-serif !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #f5af19 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.4);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #f5af19;
        text-decoration: none;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-icon">🪶</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">About KalamAI</p>', unsafe_allow_html=True)
    st.markdown('''
    <p class="sidebar-text">
    KalamAI uses a <strong>character-level LSTM neural network</strong> trained on classical 
    Urdu ghazals from Rekhta to generate unique poetry.
    </p>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="sidebar-title">💡 Tips</p>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="tip-card">
        <strong>Starting Word</strong><br>
        Use Urdu/Roman Urdu words like "ishq", "dil", "mohabbat" for best results.
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="tip-card">
        <strong>Temperature</strong><br>
        Lower = more focused & predictable<br>
        Higher = more creative & random
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="tip-card">
        <strong>Length</strong><br>
        150-250 characters usually produces 2-4 complete verses.
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('''
    <p class="sidebar-text" style="text-align: center; font-size: 0.8rem;">
    Made with ❤️ by <strong>Haseeb Abbas</strong>
    </p>
    ''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# Hero Section
st.markdown('<div class="hero-icon">✨🪶✨</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">KalamAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Urdu Poetry Generator • Trained on Classical Ghazals</p>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Input Section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<p class="section-header">✍️ Compose Your Poetry</p>', unsafe_allow_html=True)
    
    # Starting text input
    start_text = st.text_input(
        "Enter starting word or verse",
        value="pyar",
        placeholder="e.g., ishq, dil, mohabbat, pyar...",
        help="Enter an Urdu word in Roman script to start your poem"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sliders in two columns
    slider_col1, slider_col2 = st.columns(2)
    
    with slider_col1:
        generation_length = st.slider(
            "📏 Poetry Length",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Number of characters to generate"
        )
    
    with slider_col2:
        temperature = st.slider(
            "🎨 Creativity Level",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Lower = predictable, Higher = creative"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        generate_clicked = st.button("🪶 Generate Poetry", use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Poetry Output Section
if generate_clicked:
    with st.spinner("✨ Weaving words into poetry..."):
        generated_poetry = generate_text(
            loaded_model, 
            start_text, 
            loaded_char2idx, 
            loaded_idx2char, 
            generation_length, 
            temperature
        )
    
    # Display poetry
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="section-header">📜 Your Generated Poetry</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="poetry-output">{generated_poetry}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action buttons
        action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
        
        with action_col2:
            st.download_button(
                label="📥 Download Poetry",
                data=generated_poetry,
                file_name="kalamAI_generated.txt",
                mime="text/plain",
                use_container_width=True
            )

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="footer">
    <p>KalamAI • AI Poetry Generator • Powered by PyTorch & Streamlit</p>
    <p>Trained on classical ghazals from <a href="https://rekhta.org" target="_blank">Rekhta</a></p>
</div>
''', unsafe_allow_html=True)
