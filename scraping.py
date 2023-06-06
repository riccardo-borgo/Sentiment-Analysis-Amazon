import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

# Set display options
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_colwidth', None)  # Prevent wrapping of DataFrame

def scraper(url):
    review_title = []
    review_body = []
    review_stars = []
    reviews = {}
    i = 0
    
    URL = "https://www.amazon.it/echo-dot-2022/product-reviews/B09B8X9RGM/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    headers = {
            'authority': 'www.amazon.it',
            'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language': 'it-IT,it;en-GB,en-US;q=0.9,en;q=0.8',
        }

    while i < 50:
        webpage = requests.get(URL, headers=headers)
        soup = BeautifulSoup(webpage.content, 'html.parser')
        review_title.append(soup.select('a.review-title'))
        review_body.append(soup.select('div.a-row.review-data span.review-text'))
        review_stars.append(soup.select('div.a-row:nth-of-type(2) > a.a-link-normal:nth-of-type(1)'))
        try:
            next_url = soup.select_one('li.a-last a').get('href')
            URL = f"https://www.amazon.it{next_url}"
            i += 1
        except Exception as e:
            print(f'An error occured {e}')
        
    reviews = {
        'Title' : review_title,
        'Body' : review_body,
        'Stars' : review_stars
    }
    
    return reviews


def processing_text(reviews):
    review_title = [[element.text.replace('\n', '') for element in sublist] for sublist in review_title]
    review_body = [[element.text.replace('\n', '') for element in sublist] for sublist in review_body]
    review_stars = [[element.get('title').split()[0] for element in sublist] for sublist in review_stars] # getting only the number of stars the user put
    
    review_title = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_title]
    review_title = [[element.lower() for element in sublist] for sublist in review_title]
    
    review_body = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_body]
    review_body = [[element.lower() for element in sublist] for sublist in review_body]
    
    df = pd.DataFrame(columns = ['Title', 'Body', 'Stars'])
    
    df['Title'] = [item for sublist in review_title for item in sublist]
    df['Body'] = [item for sublist in review_body for item in sublist]
    df['Stars'] = [item for sublist in review_stars for item in sublist]
    
    df['Stars'] = [element.replace(',0', '') for element in df['Stars']]
    df['Stars'] = df['Stars'].astype(int)
    df['Title'] = df['Title'].astype(str)
    df['Body'] = df['Body'].astype(str)
    
    df.isnull().sum()
    
    # writing the dataframe into a CSV file just to do not have to scape again in case I do something wrong
    df.to_csv('data.csv', index=False)