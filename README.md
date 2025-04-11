# Social Media Sentiment Analysis Dashboard

## Overview
This project is a Social Media Sentiment Analysis Dashboard designed to analyze and visualize sentiment from Twitter and Facebook data. It leverages Natural Language Processing (NLP) techniques, machine learning, and interactive visualizations to provide insights into public opinion. The dashboard allows users to explore sentiment distributions, sentiment scores, word clouds, and perform real-time sentiment predictions on input text.

## Features

**Sentiment Distribution**:  
Visualizes sentiment categories (Positive, Negative, Neutral) as pie charts and bar charts.

**Sentiment Score Distribution**:  
Displays histograms and box plots for sentiment polarity scores.

**Word Clouds**:  
Generates word clouds for Positive, Negative, and Neutral sentiments.

**Top Words Analysis**:  
Highlights the most frequent words for each sentiment category.

**Sentiment Prediction**:  
Predicts the sentiment of user-provided text with cleaned output and compound score.

**Dynamic Platform Selection**:  
Switch between Twitter and Facebook data seamlessly.

## Tools and Libraries Used

**Data Processing**: PySpark, Pandas, NLTK (VADER, TextBlob)  
**Machine Learning**: Scikit-learn (TF-IDF, Naive Bayes)  
**Visualization**: Matplotlib, Seaborn, Plotly, WordCloud  
**Dashboard**: Dash (interactive web app)  
**Others**: Jupyter Notebook/Colab, Joblib (model saving)

## Project Structure
The project is organized into the following components:

**Data Collection**:  
Loads Twitter and Facebook datasets.

**Text Processing**:  
Cleans and preprocesses text data using tokenization, lemmatization, and stopword removal.

**Feature Engineering**:  
Extracts features using TF-IDF vectorization.

**Model Training**:  
Trains a Naive Bayes classifier for sentiment analysis.

**Visualization**:  
Generates interactive visualizations for sentiment analysis.

**Dashboard**:  
Provides an interactive web interface for exploring sentiment analysis results.

## Setup Instructions

### 1. Install Dependencies
Run the following command in your terminal or Jupyter Notebook to install all required libraries:

```bash
pip install pyspark nltk wordcloud dash plotly pandas seaborn scikit-learn joblib matplotlib
```

### 2. Download NLTK Resources
Ensure that the necessary NLTK resources are downloaded by running the following commands in your notebook:

```python
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
```

### 3. Load Data
Place your Twitter and Facebook datasets in the appropriate directory. Update the file paths in the code to match your dataset locations:

```python
twitter_path = '/path/to/twitter_data.csv'
facebook_path = '/path/to/facebook_data.csv'
```

### 4. Run the Dashboard
Execute the Python script or Jupyter Notebook containing the dashboard code. The dashboard will be accessible at:

```
http://127.0.0.1:8050/
```

## Usage

**Select Platform**:  
Use the dropdown menu to select either "Twitter" or "Facebook" data.

**Explore Visualizations**:  
View sentiment distribution as pie charts and bar charts.  
Analyze sentiment score distributions using histograms and box plots.  
Explore word clouds for Positive, Negative, and Neutral sentiments.

**Predict Sentiment**:  
Enter a sentence in the text area and click "Predict Sentiment" to see the predicted sentiment, compound score, and cleaned text.
