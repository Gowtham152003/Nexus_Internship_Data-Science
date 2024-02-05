'''DATA PREPROCESSING'''

import pandas as pd

# Load the existing CSV file
existing_csv_path = 'tweet.csv'
df = pd.read_csv(existing_csv_path)

# Define column names
column_names = ['Sentiment','Ids', 'timestamp','Flag','User','Text']  # Replace with your actual column names

# Add column names to the DataFrame
df.columns = column_names

# Save the DataFrame back to a CSV file with column names
new_csv_path = 'new_tweet1.csv'
df.to_csv(new_csv_path, index=False)


'''DATA EXPLORATION'''
import pandas as pd
df = pd.read_csv('new_tweet1.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
tweet_content = df['Text']
timestamp = df['timestamp']
sentiment_labels = df['Sentiment']

# Check unique values in the 'sentiment' column
print(df['Sentiment'].unique())

# Count the occurrences of each sentiment label
print(df['Sentiment'].value_counts())
# Display the first few tweets
print(df['Text'].head())

# Get basic statistics on the text data
print(df['Text'].describe())
# Display the first few timestamps
print(df['timestamp'].head())

# Check data type and convert if necessary
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get basic statistics on timestamps
print(df['timestamp'].describe())

'''EDA'''
import pandas as pd
import matplotlib.pyplot as plt


# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Convert timestamp to datetime with timezone information
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

# Check for any remaining missing values after conversion
print(df['timestamp'].isnull().sum())

# Extract date and time
df['date'] = df['timestamp'].dt.date

# Visualize Temporal Trends
daily_tweet_count = df['date'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
daily_tweet_count.plot(kind='line')
plt.title('Temporal Trends of Tweet Volume')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.show()

'''Sentiment Distribution'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Visualize Sentiment Distribution
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Labels')
plt.ylabel('Number of Tweets')
plt.show()

# Analyze the balance of sentiment classes
sentiment_counts = df['Sentiment'].value_counts()
total_tweets = len(df)

# Print class distribution
for sentiment, count in sentiment_counts.items():
    percentage = (count / total_tweets) * 100
    print(f"{sentiment}: {count} tweets ({percentage:.2f}%)")

'''Word cloud'''
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter

# Download the 'punkt' resource
nltk.download('punkt')

# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Assuming you have a 'text' column in your DataFrame
tweets = df['Text']

# Tokenize and clean the text
stop_words = set(stopwords.words('english'))

def process_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

# Process the text in each tweet
processed_tweets = tweets.apply(process_text)

# Flatten the list of tokens
all_tokens = [token for sublist in processed_tweets for token in sublist]

# Create a word frequency counter
word_freq = Counter(all_tokens)

# Visualize the most frequent words in a bar chart
common_words = word_freq.most_common(10)
common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 5))
plt.bar(common_words_df['Word'], common_words_df['Frequency'], color='blue')  # Specify a valid color here
plt.title('Top 10 Most Frequent Words in Tweets')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.show()


'''Sentiment trends'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Assuming you have 'timestamp' and 'Sentiment' columns in your DataFrame
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S %Z %Y', utc=True, errors='coerce')
df['date'] = df['timestamp'].dt.date

# Check for missing values
print(df[['date', 'Sentiment']].isnull().sum())

# Visualize Sentiment Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='Sentiment', data=df.groupby('date')['Sentiment'].mean().reset_index(), marker='o')

# Format the x-axis dates for better readability
date_form = DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate()

plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator

# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Assuming you have 'timestamp' and 'Sentiment' columns in your DataFrame
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S %Z %Y', utc=True, errors='coerce')
df['date'] = df['timestamp'].dt.date

# Check for missing values
print(df[['date', 'Sentiment']].isnull().sum())

# Visualize Sentiment Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='Sentiment', data=df.groupby('date')['Sentiment'].mean().reset_index(), marker='o')

# Format the x-axis dates for better readability
date_form = DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_locator(DayLocator(interval=1))  # Show ticks for every day
plt.gca().xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate()

plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv('new_tweet1.csv')

# Assuming you have a 'text' column in your DataFrame
tweets = df['Text']

# Define a function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Apply the preprocessing function to the 'text' column
df['processed_text'] = tweets.apply(preprocess_text)

# Display the original and processed text side by side
print(df[['Text', 'processed_text']].head())

'''ML'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load your preprocessed dataset (assuming 'processed_text' and 'sentiment' columns)
df = pd.read_csv('new_tweet1.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['Text'], df['Sentiment'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_features = tfidf_vectorizer.fit_transform(train_data)
test_features = tfidf_vectorizer.transform(test_data)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(train_features, train_labels)

# Predictions on the test set
predictions = model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(test_labels, predictions))


'''Features'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your preprocessed dataset (assuming 'processed_text' and 'sentiment' columns)
df = pd.read_csv('new_tweet1.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['Text'], df['Sentiment'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_features = tfidf_vectorizer.fit_transform(train_data)
test_features = tfidf_vectorizer.transform(test_data)

# Scale the features
scaler = StandardScaler(with_mean=False)  # With_mean=False to handle sparse matrix efficiently
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(train_features_scaled, train_labels)

# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names()

# Get coefficients (importance) of features
coefficients = model.coef_[0]

# Create a DataFrame to store feature names and their coefficients
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort features by absolute coefficient values to get the most important features
feature_importance_df['Absolute_Coefficient'] = feature_importance_df['Coefficient'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

# Visualize feature importance using a bar chart
top_n = 20  # You can adjust this based on how many top features you want to visualize
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'][:top_n], feature_importance_df['Absolute_Coefficient'][:top_n])
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.title('Top Features Importance for Sentiment Prediction')
plt.show()
