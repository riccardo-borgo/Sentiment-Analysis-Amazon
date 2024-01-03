# Sentiment Analysis of an Amazon Product Reviews

The goal of this project was to scrape amazon reviews directly from the website, processo them in order to find the sentiment of every text and finally build models that try to predict the sentiment.

## Part 1 - Scraping
All the projects that are related with the sentiment analysis and therefore with some text, need to start with scraping something from a website. In my case in order to generate a dataset full of reviews I needed to scrape the Amazon website. This was quite difficult since Amazon is a very well known website and very well protected from "attacks" like this one.

In order to retreive all the data I decided to use **selenium** as the scrape library since (maybe due to my connections or setting limitations) it was the only one that worked. Another important fact to underline is that from about June/July 2023 Amazon started to block the users to see reviews from the tenth page on, even when directly accessing from the website. This seems to be a problem that cannot be overcome so I delusionally scraped only the first 10 pages.

After the scraping I ended up with a dataframe containing:
- Title of the review
- Body of the review
- Star rating
- Verified Purchase
- Place date

And finally I stored everything into a JSON file.

## Part 2 - Sentiment Analysis
In this part of the analysis I started, first of all, managing the dataset in such a way I could work better with it. I performed some preprocessing, removing stopwords and other cleanings in order to end up with a clear review. In the end I simply calculated a sentiment based on the stars the user gave to the product. After that I performed EDA and showed plots in order to highlight some insights and shows nice things. Here there are some examples:

![image](https://github.com/riccardo-borgo/Sentiment-Analysis-Amazon/assets/51230348/f4e97fd5-f006-47d9-94cf-a66a8ec48146)

![image](https://github.com/riccardo-borgo/Sentiment-Analysis-Amazon/assets/51230348/12b6ad4c-cbf1-431f-a560-696904cecca3)

![image](https://github.com/riccardo-borgo/Sentiment-Analysis-Amazon/assets/51230348/0c7fffc0-ec2f-4199-b849-b89aded96ba7)

Then I tried to compare the sentiment resulted from the stars with other two methods, that theoretically, could overperform: **VADER** and **BERT**. And in the end it seems like that **BERT** is the most robust one.

The last part of this notebook was related to the transformation of the dataset into numbers, since ML models cannot work directly with words. The method I decided to use is **Doc2Vec** that seems more interesting rather than **TFIDF Vectorizer**.

## Part 3 - Models
The last part of the project was trying to build models to predict the seniment. Since this is a classification problem I choose the most famous one about this task:
- Naive Bayes
- Decision Tree
- Random Forest
- Logistic Regression

This is the table with the results:

Model Name | Accuracy | Recall | F1 Score 
--- | --- | --- | --- 
Näive Bayes | 0.80 | 0.80 | 0.80
Decision Tree |	0.60 |	0.60	| 0.60
Random Forest |	0.80 |	0.80 |	0.80
Logistic Regression |	0.75 |	0.75 |	0.75
