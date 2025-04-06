import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import sys

class ArticleAnalyzer:
    def __init__(self):
        """Initialize NLP models"""
        self.nlp = spacy.load("ro_core_news_lg")
        self.vectorizer = TfidfVectorizer()

    def preprocess_text(self, text):
        """Clean and lemmatize text"""
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts using TF-IDF"""
        texts = [text1, text2]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def find_similar_articles(self, input_text, articles, is_bad=False, threshold=0.1):
        """Find articles similar to input text"""
        results = []
        preprocessed_input = self.preprocess_text(input_text)
        
        for article in articles:
            # Combine title and description for better matching
            article_text = f"{article.get('title', '')} {article.get('description', '')}"
            if not article_text:
                continue
                
            preprocessed_article = self.preprocess_text(article_text)
            similarity = self.calculate_similarity(preprocessed_input, preprocessed_article)
            
            if similarity > threshold:
                results.append({
                    'title': article.get('title', ''),
                    'link': article.get('link', ''),
                    'description': article.get('description', ''),
                    'similarity': similarity,
                    'is_bad': is_bad
                })
        
        return results

def main():
    """Main function to analyze input text against articles"""
    if len(sys.argv) < 2:
        print("Usage: python article_analyzer.py 'text to analyze'")
        return
        
    input_text = ' '.join(sys.argv[1:])
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    good_articles_path = os.path.join(current_dir, 'good_articles.json')
    bad_articles_path = os.path.join(current_dir, 'bad_articles.json')
    
    good_articles = []
    bad_articles = []
    
    try:
        with open(good_articles_path, 'r', encoding='utf-8') as f:
            good_articles = json.load(f)
        print(f"Loaded {len(good_articles)} good articles")
    except Exception as e:
        print(f"Error loading good articles: {e}")
    
    try:
        with open(bad_articles_path, 'r', encoding='utf-8') as f:
            bad_articles = json.load(f)
        print(f"Loaded {len(bad_articles)} bad articles")
    except Exception as e:
        print(f"Error loading bad articles: {e}")
    
    if not good_articles and not bad_articles:
        print("No articles found to analyze")
        return
    
    # Initialize analyzer and find similar articles
    analyzer = ArticleAnalyzer()
    good_matches = analyzer.find_similar_articles(input_text, good_articles, is_bad=False)
    bad_matches = analyzer.find_similar_articles(input_text, bad_articles, is_bad=True)
    
    # Sort matches separately
    good_matches.sort(key=lambda x: x['similarity'], reverse=True)
    bad_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"\nAnalizăm textul: '{input_text}'")
    print(f"\nAm găsit {len(good_matches) + len(bad_matches)} articole relevante:")
    
    if good_matches:
        print("\nArticole de informație verificată:")
        for idx, article in enumerate(good_matches, 1):
            print(f"\n{idx}. {article['title']}")
            print(f"   Similaritate: {article['similarity']:.2f}")
            print(f"   Link: {article['link']}")
            print(f"   Descriere: {article['description']}")
    
    if bad_matches:
        print("\nArticole de dezinformare:")
        for idx, article in enumerate(bad_matches, 1):
            print(f"\n{idx}. {article['title']}")
            print(f"   Similaritate: {article['similarity']:.2f}")
            print(f"   Link: {article['link']}")
            print(f"   Descriere: {article['description']}")
    
    if not good_matches and not bad_matches:
        print("\nNu am găsit articole similare.")

if __name__ == "__main__":
    main() 