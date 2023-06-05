# Sentiment Analysis Amazon Product

The aim of this assignement was to scrape reviews from an e-coomerce website and apply the sentiment analysis on the result. After done that you have been be asked to apply some Machine Learning models, trying to predict the class: "**Bad**", "**Neutral**", "**Good**"

 ## CODE - PART 1: SCRAPING
First of all I started importing all the libraries I thought would be important and useful for the purpose of the task.
```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
```

After that I set out few rules with Pandas library in order to have a better view of the future DataFrame with all the reviews
```python
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_colwidth', None)  # Prevent wrapping of DataFrame
```

Then I delcared the variables I will use to store the different parts of the reviews and a simple index to iterate over the pages
```python
review_title = []
review_body = []
review_stars = []
i = 0
```

The next cell include the all the various steps of the scraping: 
1. Creating a variable "URL" with the first review page
2. Creating an **header** to handle request to the website: this is mandatory, otherwise we cannot access the code of a website
3. While loop that iterate the first 50 pages. This just to be sure to not exceed the maximum amount of request and being banned and also to be consistent with the results
4. Inside the while I look for all the **CSS selectors** that contains the **title**, the **body** and the **amount of stars** the person gave to the product of the review
5. The last part (very important) allow the scraper to change the URL **automatically** to the next page
```python
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
    next_url = soup.select_one('li.a-last a').get('href')
    URL = f"https://www.amazon.it{next_url}"
    i += 1
```

After done that I will have three list with all the titles, the bodies and the stars of the first 50 pages of amazon reviews. They are list of lists since every element contains 10 reviews (the ones that are in a single page).

Now it is time for a little pre-processing before creating a Data Set:
```python
review_title = [[element.text.replace('\n', '') for element in sublist] for sublist in review_title] # removing all the "\n" in the title
review_body = [[element.text.replace('\n', '') for element in sublist] for sublist in review_body] # removing all the "\n" in the bodies
review_stars = [[element.get('title').split()[0] for element in sublist] for sublist in review_stars] # getting only the number of stars the user put

review_title = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_title] # keeping only letters in the titles
review_title = [[element.lower() for element in sublist] for sublist in review_title] # converting all the text to lower case

review_body = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_body] # keeping only letters in the bodies
review_body = [[element.lower() for element in sublist] for sublist in review_body] # converting all the text to lower case
```

The need to lower everything is done majorly since it is a common practice performing sentiment analysis just be have more a "flat" text.

Now we can pass to the creation of the **Data Set** and a very short check for NA's:
```python
df = pd.DataFrame(columns = ['Title', 'Body', 'Stars'])

df['Title'] = [item for sublist in review_title for item in sublist]
df['Body'] = [item for sublist in review_body for item in sublist]
df['Stars'] = [item for sublist in review_stars for item in sublist]

df['Stars'] = [element.replace(',0', '') for element in df['Stars']]
df['Stars'] = df['Stars'].astype(int)
df['Title'] = df['Title'].astype(str)
df['Body'] = df['Body'].astype(str)

df.isnull().sum()

df.to_csv('data.csv', index=False)
```

The last command has been done just for a worries of mine, since sometimes I got banned from scraping from Amazon so I decided to put everything into a file just be sure to have all saved and uploading it when necessary

## CODE - PART 2: SENTIMENT ANALYSIS
Now It comes the interesting part. First of all I started with a very simple EDA with just few plots representing the most interesting thing when doing Sentiment Analysis: the most frequent words, in this case presented in two ways: 
1. With a barplot that represents the frequencies of the 10 most common words in titles and bodies;
2. With a Word Plot: that is a special kind of frequency plot that shows the most used word bigger in resepct to the others

To find the most common words I must **tokenize** every text. The first 4 lines of code do that. The second and the fourth, particularly, find all the **stopwords** in the text and delete them. The stopwords are words that are filtered out before, or after, the actual words in the text that carry the information. In fact, most stop words have no particular meaning when isolated from the text. Then, with the function ```.FreDist(string)``` I create a dictionary as per keys the words and as values the number of times that word occurs.

```python
list_title = df['Title'].to_list()
stopwords = nltk.corpus.stopwords.words('italian')
words_title = [word for text in list_title for word in nltk.word_tokenize(text)]
words_title_clear = [w for w in words_title if w not in stopwords]

fd_title = nltk.FreqDist(words_title_clear)

sorted_fd_title = dict(sorted(fd_title.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(15,8))
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.title('Frequency of the first ten words in the titles')
plt.bar(list(sorted_fd_title.keys())[:10], list(sorted_fd_title.values())[:10])
plt.show()
```
![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/582567c8-7dea-4194-bdf9-be234bc4102a)

```python
list_body = df['Body'].to_list()
list_body = [str(word) for word in list_body]
words_body = [word for text in list_body for word in nltk.word_tokenize(text)]
words_body_clear = [w for w in words_body if w.lower() not in stopwords]
words_body_clear = [char for char in words_body_clear if char.isalpha()]

fd_body = nltk.FreqDist(words_body_clear)

sorted_fd_body = dict(sorted(fd_body.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(15,8))
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.title('Frequency of the first ten words in the bodies')
plt.bar(list(sorted_fd_body.keys())[:10], list(sorted_fd_body.values())[:10])
plt.show()
```
![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/ea7395e1-2889-4374-98c1-38ac90bf67ff)

Now It comes the Word plots. Here I decided to plot only the good and neutral ones, since bad reviews were basically equal (semantically talking) to the neutrals.

```python
positive_reviews = df[df['Stars'] >= 4]['Title']
positive_reviews = ".".join(positive_reviews)

wordcloud = WordCloud(background_color="white", max_words=len(positive_reviews))

wordcloud.generate(positive_reviews)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```
![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/fb8d1574-7f6f-4084-933d-5ae6c4b29c0b)

```python
neutral_reviews = df[df['Stars'] == 3]['Title']
neutral_reviews = ".".join(neutral_reviews)

wordcloud = WordCloud(background_color="white", max_words=len(neutral_reviews))

wordcloud.generate(neutral_reviews)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```
![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/a808f4e0-3592-4afe-a2e4-16847cd131c4)

Now It's time to start classify the phrases as "Good", "Bad", "Neutral". 

First of all I create a sentiment analyzer object and I calculate the **polarity** of every single review title and body. 

The polarity of a text is a score (generally from -1 to 1) that explains how good, bad or neutral that phrase is. In this specific case, the method ```.polarity_scores(string)``` returns a dictionary with all the three polarities plus the compound value, that is a sort of weighted average between all of the classifications.

```python
sia = SentimentIntensityAnalyzer()
title_polarity = [sia.polarity_scores(str(element)) for element in df['Title']]
body_polarity = [sia.polarity_scores(str(element)) for element in df['Body']]
```

After done that I decided to take only the compound value of the polarity scores but the problem now was that all the reviews that would have had a neutral score of 1 were result in a compund value of polarity equal to 0. So, in order to do not have to play with zeroes (that is never good), I assigned a value of 0.5 as polarity for that specific texts.

```python
title_pol_mean = []

for i in range(len(title_polarity)):
    if title_polarity[i]['compound'] == 0.0:
        title_pol_mean.append(0.5)
    else:
        title_pol_mean.append(title_polarity[i]['compound'])
 
body_pol_mean = []

for i in range(len(body_polarity)):
    if body_polarity[i]['compound'] == 0.0:
        body_pol_mean.append(0.5)
    else:
        body_pol_mean.append(body_polarity[i]['compound'])
```

After getting all the polarities of the titles and the bodies I decided to compute a weighted average of the different polarity scores of titles and bodies. I use as weight 0.2 (20% for the titles) and 0.8 (80%) for the bodies. Usually someone could think that should be the opposite since the titled explains in a better and shorter way the sentiment of a phrase, but since I resulted with mainly only neutral titles I decided to do the opposite.

```python
total_polarity = []
total_polarity = [round((i*0.2 + j*0.8)/2, 4) for i,j in zip(title_pol_mean, body_pol_mean)] 

df['Polarity'] = total_polarity
```

Then I assigned a value among "Bad", "Neutral" and "Good" according to the value of the polarity:

```python
list_polarity = []
for i in range(len(df)):
    if df['Polarity'][i] < 0:
        list_polarity.append('Bad')
    elif df['Polarity'][i] <= 0.15:
        list_polarity.append('Neutral')
    else:
        list_polarity.append('Good')

df['Polarity_Text'] = list_polarity

df['Polarity_Text'].value_counts()
```
> Good       406<br>Bad         65<br>Neutral     39<br>Name: Polarity_Text, dtype: str

Storing the new dataset into a new CSV file:

```python
df.to_csv('data_final.csv', index=False)
```

Now, the last part of the sentiment analysis is to create a proper dataframe to feed our models. Since the standard one with all the words couldn't be admissible since Machine Learning Models doesn't know how to interpret a certain word in order to classify a phrase I needed to map every review according to the word it has in it.

To do so there is a particular module of ```sklearn.feature_extraction.text```, ```TfidfVectorizer ``` that allows us to create a dense matrix in order to see for every word a level of beloging to every review:

```python
df['Full Review'] = df.apply(lambda row: row['Title'] + ' ' + row['Body'], axis=1)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['Full Review'])
feature_names = vectorizer.get_feature_names_out()
matrix = vectors.todense()
list_dense = matrix.tolist()
sparse_matrix = pd.DataFrame(list_dense, columns=feature_names)

sparse_matrix['Polarity_Text'] = list_polarity
```

## CODE - PART 3: MACHINE LEARNING MODELS
For the last part of the code I applied some classifications models to my dataset. The one that I choose are:
1. Näive Bayes Classiers
2. Decision Tree
3. Random Forest
4. K-Neigherest Neighbors


















