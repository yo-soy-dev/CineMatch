# 🎬 CineMatch - Content-Based Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-ready movie recommendation engine with FastAPI backend and Streamlit frontend**

This end-to-end machine learning project demonstrates how to build and deploy a content-based movie recommendation system similar to those used by Netflix and Prime Video. Built using advanced NLP techniques with a FastAPI backend and interactive Streamlit interface.

---

## 🌟 Key Highlights

- **Intelligent Recommendations**: Suggests movies based on genres, cast, keywords, and plot similarity
- **Advanced NLP Pipeline**: TF-IDF vectorization and cosine similarity for accurate matching
- **Dual Interface**: FastAPI REST API + Streamlit web UI
- **Pre-computed Models**: Optimized with pickle files for instant recommendations
- **Production-Ready**: Deployed backend with automatic API documentation

---

## 🧰 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML/NLP** | scikit-learn, pandas, numpy |
| **Backend API** | FastAPI, uvicorn |
| **Frontend** | Streamlit |
| **Serialization** | pickle |
| **Data Source** | TMDB 5000 Movies (Kaggle) |

---

## 📂 Project Structure

```
movie-recommender/
│
├── .gitignore                      # Git ignore file
├── app.py                          # Streamlit frontend application
├── main.py                         # FastAPI backend server
├── requirements.txt                # Python dependencies
│
├── movies_metadata.csv             # Movie dataset (TMDB)
│
├── df.pkl                          # Processed movie dataframe
├── indices.pkl                     # Movie title to index mapping
├── tfidf.pkl                       # Trained TF-IDF vectorizer
└── tfidf_matrix.pkl                # Precomputed TF-IDF matrix
```

---

## 📊 Dataset Overview

**Source**: [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

**Features Used**:
- `title` – Movie name
- `genres` – Movie categories
- `overview` – Plot description
- `keywords` – Thematic tags
- `cast` – Main actors
- `crew` – Director info

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/devanshtiwari/movie-recommender.git
cd movie-recommender
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Pickle Files
Ensure these files are present:
- `df.pkl` – Processed movie data
- `indices.pkl` – Title-to-index mapping
- `tfidf.pkl` – TF-IDF vectorizer
- `tfidf_matrix.pkl` – Similarity matrix

*(If missing, you'll need to run preprocessing to generate them)*

---

## 🚀 Running the Application

### Option 1: Streamlit Frontend (Recommended for Users)

```bash
streamlit run app.py
```

**Access**: `http://localhost:8501`

**Features**:
- Interactive movie search
- Visual recommendation cards
- Real-time similarity scores
- User-friendly interface

---

### Option 2: FastAPI Backend (For API Integration)

```bash
uvicorn main:app --reload
```

**Access**: 
- API: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 📡 API Usage

### Endpoint: Get Recommendations

**Request**:
```http
GET /recommend?movie_title=Inception&top_n=5
```

**Response**:
```json
{
  "query_movie": "Inception",
  "recommendations": [
    {
      "title": "Interstellar",
      "similarity_score": 0.85
    },
    {
      "title": "The Prestige",
      "similarity_score": 0.82
    },
    {
      "title": "Shutter Island",
      "similarity_score": 0.78
    },
    {
      "title": "Memento",
      "similarity_score": 0.76
    },
    {
      "title": "The Dark Knight",
      "similarity_score": 0.74
    }
  ]
}
```

**Error Response**:
```json
{
  "detail": "Movie 'Unknown Title' not found in database"
}
```

---

## 🧠 How It Works

### 1. **Data Preprocessing**
- Combines movie features (genres, keywords, cast, overview) into a single text field
- Cleans and normalizes text data
- Saves processed data to `df.pkl`

### 2. **Feature Extraction**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical vectors
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
```

### 3. **Similarity Computation**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix)
```

### 4. **Recommendation Generation**
```python
def get_recommendations(title, top_n=5):
    idx = indices[title]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return df['title'].iloc[movie_indices]
```

---

## 📦 Model Files Explained

| File | Purpose | Size |
|------|---------|------|
| `df.pkl` | Processed movie dataframe with all metadata | ~2-5 MB |
| `indices.pkl` | Dictionary mapping movie titles to dataframe indices | ~100 KB |
| `tfidf.pkl` | Trained TF-IDF vectorizer for transforming text | ~500 KB |
| `tfidf_matrix.pkl` | Precomputed TF-IDF matrix for all movies | ~10-50 MB |

---

## 🎨 Streamlit Frontend Features

- **Search Box**: Type and select from movie titles
- **Recommendation Cards**: Visual display with movie posters (if integrated)
- **Similarity Scores**: Shows how closely matched each recommendation is
- **Responsive Design**: Works on desktop and mobile
- **Fast Loading**: Uses precomputed similarity matrix

---

## 🔧 Configuration

### `requirements.txt`
```txt
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.0
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
python-multipart==0.0.6
```

### `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Models (if too large)
# *.pkl

# Data
*.csv

# IDE
.vscode/
.idea/
```

---

## 🔮 Future Enhancements

- [ ] **Hybrid Recommendations**: Combine content-based + collaborative filtering
- [ ] **User Ratings**: Allow users to rate movies and personalize recommendations
- [ ] **Movie Posters**: Integrate TMDB API for images
- [ ] **Advanced NLP**: Use BERT/Sentence Transformers for semantic similarity
- [ ] **Database**: Move from CSV to PostgreSQL/MongoDB
- [ ] **Docker**: Containerize for easy deployment
- [ ] **Cloud Deployment**: Deploy on AWS/GCP/Heroku
- [ ] **Authentication**: Add user login and watch history

---

## 🚀 Deployment

### Deploy FastAPI on Render/Railway
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Deploy Streamlit
```bash
streamlit run app.py --server.port $PORT
```

---

## 📚 Resources

- [TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Content-Based Filtering](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)
- [TF-IDF Explained](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

---

## 👨‍💻 Author

**Devansh Tiwari**

- 📧 Email: devanshtiwari817@gmail.com
- 💼 GitHub: https://github.com/yo-soy-dev
- 🔗 LinkedIn: https://linkedin.com/in/yo-soy-dev

---

## 🤝 Contributing

Contributions are welcome! 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Show Your Support

If this project helped you, give it a ⭐ on GitHub!

---

**Built with ❤️ using Python, FastAPI, and Streamlit**
