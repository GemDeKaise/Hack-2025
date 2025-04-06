"""
Module for managing trusted sources for fact checking.
Loads sources from a JSON configuration file.
"""

import json
import os
from typing import Dict, Any, List, Tuple
import requests
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer for Romanian
stemmer = SnowballStemmer("romanian")

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRUSTED_SOURCES_FILE = os.path.join(CURRENT_DIR, "trusted_sources.json")

def load_trusted_sources() -> Dict[str, Any]:
    """
    Loads trusted sources from the JSON configuration file.
    
    Returns:
        Dict[str, Any]: Dictionary containing trusted sources configuration
    """
    try:
        with open(TRUSTED_SOURCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Trusted sources file not found at {TRUSTED_SOURCES_FILE}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in trusted sources file at {TRUSTED_SOURCES_FILE}")
        return {}

def get_trusted_sources() -> Dict[str, Any]:
    """
    Returns the dictionary of trusted sources with their configurations.
    """
    return load_trusted_sources()

def get_source_trust_level(source_name: str) -> float:
    """
    Returns the trust level for a given source.
    
    Args:
        source_name (str): The name of the source
        
    Returns:
        float: The trust level of the source (0.0 to 1.0)
    """
    sources = load_trusted_sources()
    if source_name in sources:
        return sources[source_name]["trust_level"]
    return 0.0

def preprocess_text(text: str) -> str:
    """
    Preprocesses text for comparison.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates similarity between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    return SequenceMatcher(None, text1, text2).ratio()

def extract_keywords(text: str) -> List[str]:
    """
    Extracts keywords from text using NLTK.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of keywords
    """
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('romanian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stem words
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens

def verify_veracity(input_text: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verifies the veracity of input text using multiple techniques.
    
    Args:
        input_text (str): Text to verify
        articles (List[Dict[str, Any]]): List of articles to compare against
        
    Returns:
        Dict[str, Any]: Verification results
    """
    # Preprocess input text
    processed_input = preprocess_text(input_text)
    input_keywords = extract_keywords(processed_input)
    
    results = {
        'veracity_score': 0.0,
        'confidence': 0.0,
        'matches': [],
        'verdict': 'unknown'
    }
    
    # Analyze each article
    for article in articles:
        # Preprocess article text
        processed_article = preprocess_text(article['title'] + ' ' + article['description'])
        article_keywords = extract_keywords(processed_article)
        
        # Calculate similarity scores
        title_similarity = calculate_similarity(processed_input, article['title'])
        desc_similarity = calculate_similarity(processed_input, article['description'])
        keyword_overlap = len(set(input_keywords) & set(article_keywords)) / len(set(input_keywords) | set(article_keywords))
        
        # Calculate overall match score
        match_score = (title_similarity * 0.4 + desc_similarity * 0.4 + keyword_overlap * 0.2)
        
        if match_score > 0.3:  # Threshold for considering a match
            results['matches'].append({
                'article': article,
                'match_score': match_score,
                'title_similarity': title_similarity,
                'description_similarity': desc_similarity,
                'keyword_overlap': keyword_overlap
            })
    
    # Calculate final veracity score
    if results['matches']:
        # Sort matches by score
        results['matches'].sort(key=lambda x: x['match_score'], reverse=True)
        
        # Calculate veracity score based on best match and source trust level
        best_match = results['matches'][0]
        source_trust = get_source_trust_level('veridica')
        results['veracity_score'] = best_match['match_score'] * source_trust
        
        # Determine verdict
        if results['veracity_score'] > 0.7:
            results['verdict'] = 'true'
        elif results['veracity_score'] > 0.4:
            results['verdict'] = 'likely_true'
        elif results['veracity_score'] > 0.2:
            results['verdict'] = 'likely_false'
        else:
            results['verdict'] = 'false'
            
        # Calculate confidence
        results['confidence'] = min(1.0, len(results['matches']) * 0.2)
    
    return results

def scrape_source(source_name: str) -> Dict[str, Any]:
    """
    Scrapes a source and returns the data.
    
    Args:
        source_name (str): The name of the source to scrape
        
    Returns:
        Dict[str, Any]: Dictionary containing scraped data
    """
    sources = load_trusted_sources()
    if source_name not in sources:
        return {"error": f"Source {source_name} not found"}
    
    source = sources[source_name]
    all_articles = []
    
    try:
        # Use the direct URL for Romanian news
        base_url = "https://www.veridica.ro/stiri/romania"
        
        # Scrape all pages up to page 15
        for page in range(1, 15):  # Pages 1 to 15
            page_url = f"{base_url}?page={page}" if page > 1 else base_url
            
            # Make the request
            response = requests.get(page_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article cards
            cards = soup.find_all('div', class_='card')
            
            if not cards:  # If no cards found, we've reached the last page
                break
                
            for card in cards:
                # Get the article link and image
                image_link = card.find('a')
                if not image_link:
                    continue
                    
                link = image_link.get('href', '')
                image = image_link.find('img')
                image_url = image.get('src', '') if image else ''
                
                # Get the article title
                title_elem = card.find('h5', class_='card-title')
                if not title_elem:
                    continue
                    
                title = title_elem.find('a').text.strip() if title_elem.find('a') else ''
                
                # Get the article description
                description_elem = card.find('p', class_='card-text')
                description = description_elem.text.strip() if description_elem else ''
                
                # Get the author and date
                author_info = card.find('div', class_='col-10')
                if author_info:
                    author = author_info.find('strong').text.strip() if author_info.find('strong') else ''
                    date = author_info.find('span', class_='text-muted').text.strip() if author_info.find('span', class_='text-muted') else ''
                else:
                    author = ''
                    date = ''
                
                all_articles.append({
                    'title': title,
                    'link': link,
                    'image_url': image_url,
                    'description': description,
                    'author': author,
                    'date': date
                })
            
            print(f"Scraped page {page} - Found {len(cards)} articles")
        
        # Save all articles to a file
        with open('articles.json', 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        
        return {
            'source': source_name,
            'base_url': base_url,
            'trust_level': source['trust_level'],
            'articles': all_articles
        }
        
    except requests.RequestException as e:
        return {"error": f"Error fetching data: {str(e)}"}
    except Exception as e:
        return {"error": f"Error processing data: {str(e)}"}

if __name__ == "__main__":
    # Get user input
    input_text = input("Enter the news to check: ")
    
    # Scrape trusted sources
    sources_data = scrape_source("veridica")
    if "error" in sources_data:
        print(f"Error: {sources_data['error']}")
        exit(1)
    
    # Verify veracity
    verification_results = verify_veracity(input_text, sources_data['articles'])
    
    # Print results
    print("\nVerification Results:")
    print(f"Verdict: {verification_results['verdict']}")
    print(f"Confidence: {verification_results['confidence']:.2f}")
    
    if verification_results['matches']:
        print("\nMatching Articles:")
        for match in verification_results['matches'][:3]:  # Show top 3 matches
            print(f"\nTitle: {match['article']['title']}")
            print(f"Match Score: {match['match_score']:.2f}")
            print(f"Link: {match['article']['link']}")