"""
Module for scraping and saving articles to result.json
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from datetime import datetime

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(CURRENT_DIR, "result.json")

def scrape_veridica() -> List[Dict[str, Any]]:
    """Scrapes articles from Veridica website."""
    try:
        # Make the request
        response = requests.get("https://veridica.ro/category/fact-checking/", timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        for card in soup.find_all('div', class_='card'):
            if not (image_link := card.find('a')):
                continue
                
            link = image_link.get('href', '')
            image = image_link.find('img')
            image_url = image.get('src', '') if image else ''
            
            if not (title_elem := card.find('h5', class_='card-title')):
                continue
                
            title = title_elem.find('a').text.strip() if title_elem.find('a') else ''
            description = card.find('p', class_='card-text').text.strip() if card.find('p', class_='card-text') else ''
            
            articles.append({
                'title': title,
                'link': link,
                'image_url': image_url,
                'description': description,
                'source': 'veridica',
                'scraped_at': datetime.now().isoformat()
            })
        
        return articles
        
    except requests.RequestException as e:
        print(f"Error fetching data: {str(e)}")
        return []
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return []

def save_articles(articles: List[Dict[str, Any]]) -> None:
    """Saves articles to result.json file."""
    try:
        with open("result.json", 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(articles)} articles to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving articles: {str(e)}")

def load_articles() -> List[Dict[str, Any]]:
    """Loads articles from result.json file."""
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading articles: {str(e)}")
        return []

def update_articles() -> None:
    """Updates the result.json file with new articles."""
    print("Starting article update...")
    new_articles = scrape_veridica()
    if new_articles:
        save_articles(new_articles)
    else:
        print("No new articles were scraped")

if __name__ == "__main__":
    update_articles() 