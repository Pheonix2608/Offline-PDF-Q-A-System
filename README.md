#  Offline AI-Powered Document Assistant

> **An offline chatbot for querying documents using local LLMs via Ollama, LangChain, FAISS & Gradio.**  
> Fully private. No cloud. Runs locally. Supports PDFs, DOCX, TXT, and CSV.

![logo](assets/logo.png)

##  Table of Contents

- [ Features](#-features)
- [ Demo](#-demo)
- [ Installation](#️-installation)
- [ Supported File Types](#-supported-file-types)
- [ Usage](#-usage)
- [ One-Click Desktop App (Windows)](#-one-click-desktop-app-windows)
- [ Docker Support](#-docker-support)
- [ Project Structure](#️-project-structure)
- [ Future Upgrades](#-future-upgrades)
- [ Credits](#-credits)

##  Features

 Fully offline document assistant  
 Multi-format support: PDF, DOCX, TXT, CSV  
 Smart chunk-based retrieval with FAISS  
 Uses Ollama with local models (`mistral`, `llama2`, etc.)  
 Per-session memory using LangChain  
 Auto-summarizes uploaded files  
 Shows matched chunks used for each answer  
 Suggests follow-up questions using LLM  
 Exports full Q&A session as `.txt` or `.json`  
 Optional `.exe` for one-click launch  
 Optional Docker container support  
 Friendly Gradio UI with PDF previews  

##  Demo

![demo-screenshot](assets/demo.png)  
> UI includes file upload, chat interface, context highlighting, suggestions, and export buttons.

##  Installation

###  Prerequisites

- Python 3.10+
- Ollama installed & running locally (https://ollama.com/)
- Models downloaded (e.g., `mistral`, `llama2`):

```bash
ollama run mistral
```

###  Step-by-Step Setup

1. **Clone this repo:**

```bash
git clone https://github.com/your-username/pdf-ai-assistant.git
cd pdf-ai-assistant
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
python ui_gradio.py
```

##  Supported File Types

| Type | Support |
|------|---------|
| `.pdf` |  Yes(with preview) |
| `.docx` | Yes |
| `.txt` | Yes |
| `.csv` | Yes |

##  Usage

1. Upload one or more supported files  
2. Summaries are auto-generated  
3. Ask questions in chat  
4. View answer + matched chunks + follow-up suggestions  
5. Export Q&A session as `.txt` or `.json`

##  One-Click Desktop App (Windows)

1. Build executable with:

```bash
pyinstaller --onefile --windowed --icon=icon.ico main.py
```

2. Run `dist/main.exe`

 No Python or terminal needed  
 Browser auto-launch or native window (via `pywebview`)

##  Docker Support

###  Build image:

```bash
docker build -t offline-pdf-assistant .
```

###  Run:

```bash
docker run -it --rm -p 7860:7860 offline-pdf-assistant
```

Ensure Ollama is running on the host, not inside the container.

##  Project Structure

```
.
├── main.py                 # Launcher (for .exe)
├── ui_gradio.py           # Main Gradio interface
├── qa_engine.py           # LLM logic, memory, prompts
├── embedder.py            # SentenceTransformer embedding
├── retriever.py           # FAISS-based semantic search
├── pdf_parser.py          # Multi-format document reader
├── assets/
│   ├── logo.png           # Logo for UI
│   └── icon.ico           # App icon for .exe
├── requirements.txt
└── Dockerfile
```

##  Future Upgrades

- [ ] Highlight chunk directly in document view  
- [ ] Add user login / session-based dashboards  
- [ ] Export full conversation as PDF  
- [ ] Voice input + text-to-speech replies  
- [ ] Cloud sync (opt-in)

##  Credits

Made by **Aaryaksh Bhatnagar**  
B.Tech AI & DS – Arya College of Engineering & IT  
> Portfolio-ready AI product built from scratch for private LLM workflows
