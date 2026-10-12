# Sentiment Analysis using LSTM - Simple Version
import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Download stopwords if not available
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('data/tweets.csv')

# Filter required columns
df = df[['candidate', 'sentiment', 'text']]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)              # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)             # remove punctuation and numbers
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Encode sentiment labels
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

# Tokenization
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(df['clean_text'].values)
X = tokenizer.texts_to_sequences(df['clean_text'].values)
X = pad_sequences(X)

y = pd.get_dummies(df['sentiment_encoded']).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  # 3 classes for sentiment

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model
model.save('models/sentiment_model.h5')
print("✅ Model saved successfully in 'models/sentiment_model.h5'")
