<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-green.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<h1 align="center">✨🪶 KalamAI 🪶✨</h1>

<p align="center">
  <strong>AI-Powered Urdu Poetry Generator • Trained on Classical Ghazals from Rekhta</strong>
</p>

<p align="center">
  <em>Generate beautiful Urdu poetry using a character-level LSTM neural network trained on thousands of classical ghazals.</em>
</p>

---

## 🌟 Features

- **🎭 AI Poetry Generation**: Generate unique Urdu poetry in Roman script using deep learning
- **📚 Trained on Classical Ghazals**: Model trained on a rich corpus of ghazals scraped from [Rekhta](https://rekhta.org)
- **🎨 Beautiful Web Interface**: Modern, glassmorphism-styled Streamlit app with dark theme
- **⚡ Real-time Generation**: Instant poetry generation with customizable parameters
- **📥 Download Feature**: Save your generated poetry as a text file
- **🎛️ Adjustable Controls**: Fine-tune creativity (temperature) and poetry length

## 🏗️ Architecture

KalamAI uses a **Character-level Recurrent Neural Network (CharRNN)** with the following architecture:

| Component | Specification |
|-----------|---------------|
| **Model Type** | LSTM (Long Short-Term Memory) |
| **Embedding Size** | 128 dimensions |
| **Hidden Size** | 256 units |
| **Layers** | 2 stacked LSTM layers |
| **Dropout Rate** | 0.3 (for regularization) |
| **Sequence Length** | 256 characters |

## 📦 Project Structure

```
KalamAI/
├── app.py                    # Streamlit web application
├── Training.ipynb            # Model training notebook (Google Colab)
├── ScrapingCode.ipynb        # Data scraping notebook for Rekhta
├── my_char_rnn_model.pth     # Pre-trained model weights
├── updated_ghazals.txt       # Training dataset (~486KB of ghazals)
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages (for deployment)
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/KalamAI.git
   cd KalamAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 💻 Usage

### Web Interface

1. **Enter a Starting Word**: Type an Urdu word in Roman script (e.g., "ishq", "dil", "mohabbat", "pyar")
2. **Adjust Poetry Length**: Use the slider to set the number of characters (50-500)
3. **Set Creativity Level**: 
   - Lower values (0.1-0.5): More focused and predictable output
   - Higher values (0.6-1.5): More creative and varied output
4. **Click "Generate Poetry"**: Watch as AI creates unique verses
5. **Download**: Save your generated poetry as a text file

### Tips for Best Results

| Setting | Recommendation |
|---------|----------------|
| **Starting Word** | Use common Urdu poetry words like "ishq", "dil", "mohabbat" |
| **Temperature** | 0.7 works well for balanced creativity |
| **Length** | 150-250 characters typically produces 2-4 complete verses |

## 🔬 Training Your Own Model

### Data Collection

The `ScrapingCode.ipynb` notebook scrapes ghazals from Rekhta:

```python
# Run the scraper to collect ghazals
python -m jupyter notebook ScrapingCode.ipynb
```

### Model Training

Use `Training.ipynb` on Google Colab (recommended for GPU access):

1. Upload `updated_ghazals.txt` to Colab
2. Run all cells in `Training.ipynb`
3. Download the trained model (`best_char_rnn_model.pth`)

**Training Features:**
- Early stopping with patience of 5 epochs
- Learning rate scheduling (ReduceLROnPlateau)
- L2 regularization (weight decay)
- 80/20 train/validation split
- Adam optimizer with learning rate 0.003

## 📊 Sample Output

```
Input: "pyar"

Generated:
pyar shahi mein wo patthar ho
'ais hatheli mein ek rishte main ne dekhte hain
roz maidan-e-jang lagta hai
meri aankhon mein ye nami kyon hai
jis ko aankho se dur rakhna tha
aaj qurbat mein phir wahi kyon
```

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `torch` | Deep learning (PyTorch) |
| `numpy` | Numerical operations |
| `Pillow` | Image processing |

## 🌐 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Rekhta](https://rekhta.org)** - For the beautiful collection of Urdu ghazals
- **PyTorch** - For the deep learning framework
- **Streamlit** - For the amazing web app framework

## 👨‍💻 Author

**Haseeb Abbas**

---

<p align="center">
  Made by Haseeb Abbas
</p>
