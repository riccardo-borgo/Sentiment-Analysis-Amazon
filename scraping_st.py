import streamlit as st
import pandas as pd
import numpy as np
from function_scraping import retreive_products

st.set_page_config(layout='wide')

st.title('Amazon Product Scraper')

st.text('Insert the link of the product you want the review of: ')
URL = st.text_input(label='URL', max_chars=255, label_visibility='hidden')

if st.button("Submit"):
    url = URL
else:
    url = None

retreive_products(url)

