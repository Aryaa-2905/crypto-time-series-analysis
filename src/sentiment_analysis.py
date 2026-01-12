from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Sample crypto news headlines (can be replaced later)
news = [
    "Bitcoin price surges as institutional investors show interest",
    "Crypto market crashes amid global economic uncertainty",
    "Bitcoin adoption increases among major companies",
    "Regulatory concerns cause fear in crypto markets",
    "Positive sentiment grows as Bitcoin hits new highs"
]

# Analyze sentiment
sentiments = []
for headline in news:
    blob = TextBlob(headline)
    sentiments.append(blob.sentiment.polarity)

# Create DataFrame
df = pd.DataFrame({
    "Headline": news,
    "Sentiment Score": sentiments
})

# Print results
print(df)

# Plot sentiment
plt.figure(figsize=(8,4))
plt.plot(df["Sentiment Score"], marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("Crypto News Sentiment Analysis")
plt.ylabel("Sentiment Score")
plt.xlabel("News Index")
plt.show()
