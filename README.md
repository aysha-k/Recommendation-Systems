
# 🎬 Movie Recommendation System

This project is a movie recommendation system built using Python and key data science libraries such as **Pandas**, **Scikit-learn**, and **Cosine Similarity**. The goal is to suggest similar movies based on the movie description (overview) using TF-IDF vectorization.

## 📁 Dataset

The system uses the **TMDB 5000 Movie Dataset**, which includes movie metadata like titles, overviews, genres, cast, crew, and more. You can find the original dataset on [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

## 🚀 Features

- Preprocessing of movie overviews
- TF-IDF Vectorization to convert text to numerical data
- Cosine Similarity to compute similarity scores between movies
- Recommendation function that suggests similar movies based on the given title

## 📌 Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Numpy**
- **Jupyter Notebook**

## 🧠 How It Works

1. **Text Cleaning**: The overview column (movie description) is cleaned by filling in missing values and removing stopwords.
2. **TF-IDF Vectorization**: Converts each overview into a vector based on word importance (term frequency–inverse document frequency).
3. **Cosine Similarity**: Measures how similar two movie vectors are by calculating the cosine of the angle between them.
4. **Recommendation Function**: Given a movie title, it returns the top N most similar movies.

## 📈 Output Example

The TF-IDF matrix output shape:
```
(4800, 20978)
```
This means there are 4800 movie overviews represented using 20,978 unique words.

## 🔍 Sample Code Snippet

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize movie overviews
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## 📥 How to Run

1. Clone the repository
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and run all cells

## 📌 Next Steps

- Incorporate genres, cast, and keywords for hybrid recommendations
- Deploy the model using Flask or Streamlit
- Add user-based collaborative filtering

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

