## YT Brain — YouTube Content Strategist

An interactive Streamlit app to research, analyze, and strategize YouTube content. It fetches trending and high-engagement videos, analyzes comments and keywords, generates AI-driven recommendations, and exports polished PDF reports. Includes a basic user system for gated access.

### Key Features
- **Topic Intelligence**: Fetches videos by recent, most viewed, trending, and most liked; computes engagement metrics.
- **Single URL Deep-Dive**: Analyze any YouTube video for performance, content keywords, hashtags, and actionable recommendations.
- **AI Strategy**: Uses Google Generative AI for tailored content strategy suggestions.
- **NLP & Keywords**: VADER sentiment, KeyBERT keyword extraction, simple hashtag/keyword parsing.
- **Visualizations**: Charts with Matplotlib/Seaborn.
- **Reports**: Exports concise PDF reports (topic and URL analyses).
- **Media Utilities**: Optional audio download for a given URL (via `yt-dlp`).
- **User Access**: Basic authentication stored in `users.json` (bcrypt-hashed).

### Tech Stack
- Python, Streamlit
- YouTube Data API v3
- Google Generative AI
- Pandas, NLTK (VADER), KeyBERT, sentence-transformers, scikit-learn
- Matplotlib, Seaborn, Pillow
- ReportLab for PDF
- yt-dlp for media

---

## Project Structure

```
NewProject-YT/
  P1/
    app.py                   # Streamlit app entrypoint
    youtube_data_handler.py  # YouTube API fetch and processing utils
    requirements.txt         # Python dependencies
    users.json               # User store (created/updated at runtime)
```

---

## Prerequisites
- Python 3.9+ recommended
- A YouTube Data API v3 key
- A Google Generative AI key

Note: The app currently references API keys directly in code for convenience. For production, move these to environment variables or a secrets manager.

---

## Setup

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Configure API Keys

You need two keys:
- YouTube Data API v3 key
- Google Generative AI key

Update these in code or set env vars and modify the files to read them:
- `P1/app.py`: `genai.configure(api_key=...)` and `YOUTUBE_API_KEY = ...`
- `P1/youtube_data_handler.py`: `YOUTUBE_API_KEY = ...`

Example (temporary) in code:

```python
# in P1/app.py
genai.configure(api_key="YOUR_GEMINI_KEY")
YOUTUBE_API_KEY = "YOUR_YT_API_KEY"

# in P1/youtube_data_handler.py
YOUTUBE_API_KEY = "YOUR_YT_API_KEY"
```

Preferably, switch these to read from environment variables for safety.

4) First run will download some NLTK data automatically (quiet mode). Sentence-transformers and Torch can be large; the first install may take time.

---

## Run the App

From the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

---

## Login & Users
- On first run, the app will create `users.json` if missing and add a default user:
  - username: `admin`
  - password: `admin123`
- You can edit users within the app workflow or by modifying `users.json` directly (passwords must be bcrypt-hashed; use the app helpers to avoid manual hashing).

---

## How It Works (High-Level)
- `youtube_data_handler.py` queries YouTube Search and Videos endpoints, paginates with rate limiting, filters very short videos, and computes engagement signals. It also fetches top comments for sampled videos.
- `app.py` orchestrates Streamlit pages for:
  - Topic-based research and comparisons
  - Single URL performance analysis
  - Keyword/hashtag extraction and sentiment overview
  - PDF report generation and optional media utilities

---

## Configuration Tips
- API Quotas: Heavy use will consume YouTube API quota quickly. Expect 403 (quota exceeded) errors if exceeded; the app handles some cases gracefully.
- Regional/blocked content or comments may limit analyses.
- For production: externalize secrets, secure `users.json`, and consider replacing built-in auth with a proper provider.

---

## Troubleshooting
- Install slowness/timeouts: `torch` and `sentence-transformers` are large. Consider commenting `torch` in `requirements.txt` if you do not need KeyBERT/transformers features.
- SSL or HTTP errors on media: retry or ensure `yt-dlp` is updated.
- YouTube 403 errors: verify API key validity and quota; reduce max results.
- Streamlit not opening browser: copy the URL printed in the terminal into your browser.

---

## Development Notes
- Code style: standard Python; Streamlit app is a single entrypoint (`P1/app.py`).
- Linting/formatting are not enforced in this project by default.
- Consider refactoring API keys to environment variables and adding a `.streamlit/secrets.toml` for Streamlit.

---

## License
This project is provided as-is. Add a license here if you plan to distribute.



