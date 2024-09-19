import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob

# Load data
@st.cache
def load_data():
    data = pd.read_csv('reviews.csv')
    data_clean = data.dropna().drop_duplicates()
    return data_clean

# Function to train model
def train_models(data_clean):
    X = data_clean['Review']  # Features (reviews)
    y = data_clean['Overall_Rating']  # Target (rating)

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to numerical features using Tfidf
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_vec.toarray())
    X_test_scaled = scaler.transform(X_test_vec.toarray())

    # Train logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)

    # Train decision tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_scaled, y_train)
    y_pred_tree = decision_tree.predict(X_test_scaled)

    return logreg, decision_tree, vectorizer, scaler

# Sentiment analysis function
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Visualize top cities
def plot_top_cities(data_clean):
    top_cities = data_clean['City'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    top_cities.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Cities with Most Restaurants')
    plt.ylabel('Number of Restaurants')
    plt.xlabel('City')
    st.pyplot(plt)

# Filter restaurants by rating and price
# Function to filter restaurants based on rating and price
def filter_restaurants(data_clean, rating, price):
    # Convert 'Overall_Rating' and 'Rate for two' columns to numeric
    data_clean['Overall_Rating'] = pd.to_numeric(data_clean['Overall_Rating'], errors='coerce')
    data_clean['Rate for two'] = pd.to_numeric(data_clean['Rate for two'], errors='coerce')

    # Drop rows with NaN values in 'Overall_Rating' and 'Rate for two'
    data_clean = data_clean.dropna(subset=['Overall_Rating', 'Rate for two'])

    # Filter data based on the provided rating and price range
    filtered_data = data_clean[(data_clean['Overall_Rating'] >= rating) & (data_clean['Rate for two'] <= price)]
    
    return filtered_data


# Streamlit app layout
st.title('Restaurant Rating Prediction, Sentiment Analysis, and Restaurant Finder')

# Load data
data_clean = load_data()

# Show dataset
if st.checkbox("Show raw data"):
    st.write(data_clean.head())

# Train models
logreg, decision_tree, vectorizer, scaler = train_models(data_clean)

# Make predictions and sentiment analysis
review_input = st.text_area("Enter a review to predict the rating and analyze sentiment:")

if review_input:
    review_vec = vectorizer.transform([review_input])
    review_scaled = scaler.transform(review_vec.toarray())

    # Logistic Regression Prediction
    logreg_pred = logreg.predict(review_scaled)
    st.write(f"Predicted Rating (Logistic Regression): {logreg_pred[0]}")

    # Decision Tree Prediction
    decision_tree_pred = decision_tree.predict(review_scaled)
    st.write(f"Predicted Rating (Decision Tree): {decision_tree_pred[0]}")

    # Sentiment Analysis
    sentiment_score = analyze_sentiment(review_input)
    st.write(f"Sentiment Polarity: {sentiment_score}")
    if sentiment_score > 0:
        st.write("The sentiment is Positive.")
    elif sentiment_score < 0:
        st.write("The sentiment is Negative.")
    else:
        st.write("The sentiment is Neutral.")

# Visualize top cities
if st.checkbox("Show Top Cities with Restaurants"):
    plot_top_cities(data_clean)

# Filter by rating and price
st.header('Find Restaurants by Rating and Price')
rating_input = st.slider('Select minimum rating:', min_value=1, max_value=5, value=3)
price_input = st.slider('Select maximum price range (100-5000):', min_value=100, max_value=5000, value=3)

filtered_restaurants = filter_restaurants(data_clean, rating_input, price_input)
st.write(f"Restaurants with rating >= {rating_input} and price <= {price_input}:")
st.dataframe(filtered_restaurants[['Name', 'City', 'Overall_Rating', 'Rate for two']])
