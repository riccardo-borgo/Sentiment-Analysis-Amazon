import streamlit as st
import pandas as pd
import numpy as np
import time
import requests

def retreive_products(label):
    URL = "https://www.amazon.it/echo-dot-2022/product-reviews/B09B8X9RGM/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    headers = {
            'authority': 'www.amazon.it',
            'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language': 'it-IT,it;en-GB,en-US;q=0.9,en;q=0.8',
        }

    while i <= 50:
        try:
            webpage = requests.get(URL, headers=headers)
            # Process the response if the request was successful
            if webpage.status_code == 200:
                # Starting the scraping
                soup = BeautifulSoup(webpage.content, 'html.parser')
                print(f'Scraping page {i}')
                review_title.append(soup.select('a.review-title')) # css selector for the title of the review
                review_body.append(soup.select('div.a-row.review-data span.review-text')) # css selector for the body of the review
                review_stars.append(soup.select('div.a-row:nth-of-type(2) > a.a-link-normal:nth-of-type(1)')) # css selector for the stars of the review
                try:
                    next_link = soup.select_one('li.a-last a')
                    if next_link is not None:
                        next_url = next_link.get('href')
                        URL = f"https://www.amazon.it{next_url}"
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
    

st.set_page_config(layout='wide')

st.title('Amazon Product Scraper')

st.text('Insert the name of the link of the product you want the review of: ')
st.text_input(label='', max_chars=255)