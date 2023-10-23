import time
import requests
from bs4 import BeautifulSoup

def retreive_products(url):
    i = 0
    title_review = []
    body_review = []
    star_review = []

    headers = {
            'authority': 'www.amazon.it',
            'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language': 'it-IT,it;en-GB,en-US;q=0.9,en;q=0.8',
        }

    while i <= 50:
        try:
            webpage = requests.get(url, headers=headers)
            # Process the response if the request was successful
            if webpage.status_code == 200:
                # Starting the scraping
                soup = BeautifulSoup(webpage.content, 'html.parser')
                print(f'Scraping page {i}')
                title_review.append(soup.select('a.review-title')) # css selector for the title of the review
                body_review.append(soup.select('div.a-row.review-data span.review-text')) # css selector for the body of the review
                star_review.append(soup.select('div.a-row:nth-of-type(2) > a.a-link-normal:nth-of-type(1)')) # css selector for the stars of the review
                try:
                    next_link = soup.select_one('li.a-last a')
                    if next_link is not None:
                        next_url = next_link.get('href')
                        url = f"https://www.amazon.com{next_url}"
                except Exception as e:
                    print(f'An error occured {e}')
            else:
                # Handle the response if it's not successful
                print(f"Request failed with status code: {webpage.status_code}")
        except requests.RequestException as e:
            # Handle any exceptions that occur during the request
            print(f"An error occurred: {e}")
        i += 1
        time.sleep(2)
        
        return title_review, body_review, star_review

def text_processing(title_review, body_review, star_review):
    pass

def final_check_writing(data):
    pass