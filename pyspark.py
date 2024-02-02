import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from textblob import TextBlob


def preprocess_text(text):
  """Preprocesses text by removing unwanted characters and applying lowercase."""
  text = text.lower()
  text = text.replace('http\S+', '') # Remove URLs
  text = text.replace('@\w+', '') # Remove mentions
  text = text.replace('#', '') # Remove hashtags
  text = text.replace('RT', '') # Remove retweets
  text = text.replace(':', '') # Remove colons
  return text

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
def predict_sentiment(text):
  """Predicts sentiment using the provided transformer model."""
  # Load model and tokenizer (ensure library installation)

  # Tokenize and encode text
  tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

  # Perform model inference
  outputs = model(**tokens)
  probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
  predicted_class = np.argmax(probabilities)

  # Retrieve sentiment label from model configuration
  sentiment_labels = model.config.label2id
  predicted_sentiment = list(sentiment_labels)[predicted_class]

  return predicted_sentiment


def text_classification(words):
  """Applies sentiment classification using the predict_sentiment UDF.

  Args:
    words (DataFrame): DataFrame containing a 'word' column.

  Returns:
    DataFrame: DataFrame with an added 'sentiment' column indicating sentiment predictions.
  """

  # Preprocess text and create UDF
  words = words.withColumn('word', preprocess_text(words.word))
  predict_sentiment_udf = udf(predict_sentiment, StringType())

  # Apply UDF and return classified DataFrame
  words = words.withColumn("sentiment", predict_sentiment_udf(words.word))
  return words


if __name__ == "__main__":
  # Create SparkSession
  spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()

  # Read tweet data from socket (adjust host and port if needed)
  lines = spark.readStream.format("socket").option("host", "0.0.0.0").option("port", 5555).load()

  # Explode and preprocess words
  words = lines.select(explode(split(lines.value, "t_end")).alias("word"))
  words = words.withColumn('word', preprocess_text(words.word))

  # Classify text using the new function
  words = text_classification(words)

  # Write results to Parquet (adjust path and options)
  query = words.writeStream.queryName("all_tweets") \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "./parc") \
    .option("checkpointLocation", "./check") \
    .trigger(processingTime='60 seconds') \
    .start()

  query.awaitTermination()