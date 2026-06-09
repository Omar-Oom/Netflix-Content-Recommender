# 🎬 Netflix Content Recommender

A comprehensive, interactive web application built with **Streamlit** that provides Exploratory Data Analysis (EDA), Content Clustering, and a sophisticated Content-Based Recommendation System for Netflix movies and TV shows.

## 🌟 Features

### 1. Interactive Dashboard & Overview
* High-level metrics showing the total number of titles, movies, TV shows, and countries available in the dataset.
* Visual breakdowns of content added over time and the split between movies and TV shows.

### 2. Exploratory Data Analysis (EDA)
Dive deep into the Netflix catalog with interactive charts powered by Plotly:
* **Genres:** View top genres and their distribution across movies and TV shows.
* **Countries:** Explore global content production maps and top contributing countries.
* **Trends:** Track release trends over the years and content rating distributions.
* **Duration:** Analyze the duration of movies and the number of seasons for TV shows.

### 3. Content Clustering 🔵
Unsupervised machine learning to group similar content together:
* Uses **K-Means clustering** on dimensionality-reduced (TruncatedSVD) TF-IDF vectors.
* Evaluates clustering performance using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
* Interactive 2D PCA scatter plots to visualize the clusters.
* View profiles and top genres for each generated cluster.

### 4. Advanced Recommendation Engine 🎯
Find your next watch using a custom-built hybrid scoring system:
* **Single Title Mode:** Get personalized recommendations based on a specific movie or TV show.
* **Batch Mode:** Combine preferences from multiple seed titles (2-5 titles) to get averaged recommendations.
* **Customizable Weights:** Use sidebar sliders to adjust the importance of:
  * 🧠 Text Similarity (Descriptions, Cast, Director)
  * 🎭 Genre Overlap
  * 📅 Release Year Closeness
* **Explainable AI (XAI):** Don't just get a list of movies; understand *why* they are recommended. The app provides a detailed breakdown of score components, shared genres, shared cast, same directors, and cluster matching.

## 🛠️ Technology Stack

* **Frontend framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (TF-IDF, KMeans, TruncatedSVD, Cosine Similarity)
* **Data Visualization:** Plotly (Express & Graph Objects)

## 🚀 How to Run Locally

1. **Clone or download the repository.**
2. **Ensure you have Python installed.** (Python 3.8+ recommended)
3. **Install the required dependencies.** It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not available, install the main packages manually:)*
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
5. **Open your browser.** The app will automatically open in your default browser at `http://localhost:8501`.

## 📁 Project Structure

* `app.py`: The main Streamlit application script containing all UI and logic.
* `netflix_titles.csv` & `netflix_titles_cleaned.csv`: The datasets containing Netflix content information.
* `*.ipynb`: Jupyter notebooks used for data preparation, cleaning, and experimenting with the recommendation model.

## 🎨 UI/UX Design

The application features a custom CSS theme inspired by Netflix's actual user interface, complete with dark mode, red accents, glassmorphism elements, and sleek typography for a premium user experience.
