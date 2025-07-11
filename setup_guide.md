# 🧠 AI Document Assistant - Setup Guide

## Prerequisites

### 1. Install Ollama
```bash
# Visit https://ollama.com/ and download for your OS
# Or use package managers:

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from website
```

### 2. Install and Run a Model
```bash
# Install and run Mistral (recommended)
ollama run mistral

# Or try other models:
ollama run llama2
ollama run phi3
ollama run gemma
```

### 3. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr build-essential
```

#### macOS:
```bash
brew install poppler tesseract
```

#### Windows:
- Download poppler-utils from: https://github.com/oschwartz10612/poppler-windows
- Add to PATH
- Install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

## Installation Steps

### 1. Clone and Setup
```bash
git clone <your-repo>
cd pdf-ai-assistant
pip install -r requirements.txt
```

### 2. Test Ollama Connection
```bash
python qa_engine.py
```

### 3. Run the Application
```bash
python ui_gradio.py
```

## Common Issues and Solutions

### ❌ "Failed to connect to Ollama"
**Solution:**
```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull mistral
ollama run mistral
```

### ❌ "No module named 'fitz'"
**Solution:**
```bash
pip install PyMuPDF
```

### ❌ "pdf2image requires poppler"
**Solution:**
- Install poppler-utils (see system dependencies above)
- On Windows, ensure poppler bin folder is in PATH

### ❌ "ImportError: cannot import name 'OllamaLLM'"
**Solution:**
```bash
pip install --upgrade langchain-ollama
```

### ❌ "FAISS installation failed"
**Solution:**
```bash
# Try CPU version first
pip install faiss-cpu

# If you have GPU and CUDA:
pip install faiss-gpu
```

### ❌ "ModuleNotFoundError: No module named 'docx'"
**Solution:**
```bash
pip install python-docx
```

## Testing Your Setup

### 1. Test Individual Components
```bash
# Test Ollama
python -c "from langchain_ollama import OllamaLLM; llm = OllamaLLM(model='mistral'); print(llm.invoke('Hello'))"

# Test embeddings
python -c "from embedder import get_embeddings; print(len(get_embeddings(['test'])))"

# Test document parsing
python -c "from pdf_parser import extract_text_from_multiple_files; print('PDF parser OK')"
```

### 2. Test Full Application
```bash
python ui_gradio.py
```

## Performance Tips

### 1. Choose Right Model
- **Fast**: `mistral` (7B parameters)
- **Balanced**: `llama2` (7B parameters)
- **Quality**: `llama2:13b` (13B parameters, slower)

### 2. Optimize Chunk Size
Edit `pdf_parser.py`:
```python
def chunk_text(text, chunk_size=300, overlap=30):  # Smaller chunks for faster processing
```

### 3. Limit Context Length
Edit `qa_engine.py`:
```python
if len(context) > 2000:  # Reduce from 4000 for faster responses
    context = context[:2000] + "..."
```

## Docker Alternative

If you're having dependency issues:

```bash
# Build image
docker build -t pdf-assistant .

# Run (make sure Ollama is running on host)
docker run -p 7860:7860 pdf-assistant
```

## Troubleshooting Steps

1. **Check Ollama Status:**
   ```bash
   ollama list  # Should show installed models
   ollama ps    # Should show running models
   ```

2. **Test Network Connection:**
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"Hello"}'
   ```

3. **Check Python Environment:**
   ```bash
   python --version  # Should be 3.10+
   pip list | grep -E "(langchain|gradio|faiss|sentence-transformers)"
   ```

4. **Verify File Uploads:**
   - Try with a simple text file first
   - Check file permissions
   - Ensure files aren't corrupted

## Advanced Configuration

### Custom Models
Edit `qa_engine.py` to use different models:
```python
llm = OllamaLLM(model="your-model-name")
```

### Memory Management
Adjust memory settings in `qa_engine.py`:
```python
# Increase memory buffer
memory_sessions[session_id] = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
    max_token_limit=4000  # Adjust based on your needs
)
```

### UI Customization
Modify `ui_gradio.py` for custom themes:
```python
demo = gr.Blocks(theme=gr.themes.Monochrome())  # Different theme
```

## Getting Help

1. **Check Logs**: Look for error messages in terminal
2. **Test Components**: Use the individual test commands above
3. **Check Dependencies**: Ensure all requirements are installed
4. **Ollama Issues**: Visit https://ollama.com/docs for official help

## Success Indicators

✅ You should see:
- "✅ Connected to Ollama with mistral model"
- "✅ Ollama connection verified"
- Gradio interface loads at http://localhost:7860
- File uploads work without errors
- Q&A generates responses