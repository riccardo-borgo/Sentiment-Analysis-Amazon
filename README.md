# Sentiment Analysis Amazon Product

The aim of this assignement was to scrape reviews from an e-commerce (Amazon) website and apply the sentiment analysis on the result. After done that I apply the Machine Learning models I found most suitable in order to predict the class: "**Bad**", "**Neutral**", "**Good**" depending of the sentiment of a specific review.

 ## CODE - PART 1: scraping.ipynb
First of all I started importing all the libraries I thought would be important and useful for the purpose of the task.

```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import time
import matplotlib.pyplot as plt
from matplotlib.cm import Blues
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_models import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
```

Then I delcared some variables I will use to store the different parts of the reviews and a simple index to iterate over the pages:

```python
review_title = []
review_body = []
review_stars = []
i = 1
```

The next cell include all the various steps of the scraping: 
1. Creating a variable "URL" with the first review page;
2. Creating an **header** to handle request to the website: this is mandatory, otherwise we cannot access the page;
3. While loop that iterate the first 50 pages. This just to be sure to not exceed the maximum amount of request and being banned and also to be consistent with the results;
4. Inside the while I look for all the **CSS selectors** that contains the **title**, the **body** and the **amount of stars** the person gave to the product. Between two different requests I put a sleep of 3 seconds in order to avoid to be banned from requesting;
5. The last part (very important) allow the scraper to change the URL **automatically** to the next page.

```python
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
```

After done that I will have three lists with all the titles, the bodies and the stars of the first 50 pages of amazon reviews with a total of 500 reviews.

Now it is time for a little pre-processing before creating the dataset:

```python
review_title = [[element.text.replace('\n', '') for element in sublist] for sublist in review_title] # removing all the "\n" in the title
review_body = [[element.text.replace('\n', '') for element in sublist] for sublist in review_body] # removing all the "\n" in the bodies
review_stars = [[element.get('title').split()[0] for element in sublist] for sublist in review_stars] # getting only the number of stars the user put

review_title = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_title] # keeping only letters in the titles
review_title = [[element.lower() for element in sublist] for sublist in review_title] # converting all the text to lower case

review_body = [[re.sub("[^a-zA-ZÀ-ÖØ-öø-ÿ]", " ", element) for element in sublist] for sublist in review_body] # keeping only letters in the bodies
review_body = [[element.lower() for element in sublist] for sublist in review_body] # converting all the text to lower case
```

Now we can pass to the creation of the dataset and a very short check for NA's:

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
```
> Title    0<br>Body     0<br>Stars    0<br>dtype: int64

```python
df.to_csv('data.csv', index=False)
```
Here there is a sample of the dataset I've just created:

<img width="1468" alt="Screenshot 2023-06-09 alle 17 02 42" src="https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/e219ba6b-0862-4f30-9d4d-444f3dc1d4cb">

## CODE - PART 2: sentiment_analysis.ipynb
Now It comes the interesting part. First of all I started with a very simple EDA with just few plots representing the most frequent words, with a barplot.

To find the most common words I must **tokenize** every text. The first 4 lines of the code underneath do that. The second and the fourth, particularly, find all the **stopwords** in the text and delete them. The stopwords are words that are filtered out before, or after, the actual words in the text that carry the information. In fact, most stop words have no particular meaning when isolated from the text. Then, with the function ```.FreDist(string)``` I create a dictionary as per keys the word and as values the number of times that word occurs.

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

Now It's time to start classify the phrases as "Good", "Neutral" or "Bad". 

First of all I create a sentiment analyzer object and I calculate the **polarity** of every single review title and body. 

The polarity of a text is a score (generally from -1 to 1) that explains how good, neutral or bad that phrase is. In this specific case, the method ```.polarity_scores(string)``` returns a dictionary with all the three polarities plus the compound value, that is a sort of weighted average among all of the classifications.

```python
sia = SentimentIntensityAnalyzer()
title_polarity = [sia.polarity_scores(str(element)) for element in df['Title']]
body_polarity = [sia.polarity_scores(str(element)) for element in df['Body']]
```
<img width="462" alt="Screenshot 2023-06-06 alle 15 49 17" src="https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/bee6044b-bcbf-43b4-9d5c-31a3c411b0a4">

Above there is a snapshot of what ```.polarity_scores(string)``` returns.

After done that I decided to take only the compound value of the polarity scores, but the problem now was that all the reviews that have a neutral score of 1 had a compund value equal to 0. So, in order to do not have to play with zeroes (that is never good), I assigned a value of 0.5 for that specific review.

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

After getting all the polarities of the titles and the bodies I decided to compute a weighted average of the different polarity scores of titles and bodies. I use as weight 0.2 (20% for the titles) and 0.8 (80%) for the bodies. Usually someone could think that should be the opposite since the titled explains in a better and shorter way the sentiment of a phrase, but since I resulted with mainly neutral titles I did the opposite.

```python
total_polarity = []
total_polarity = [round((i*0.2 + j*0.8)/2, 4) for i,j in zip(title_pol_mean, body_pol_mean)] 

df['Polarity'] = total_polarity
```

Here I assigned a value among "Good", "Neutral" or "Bad" according to the value of the polarity:

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
> Good       406<br>Bad         66<br>Neutral     38<br>Name: Polarity_Text, dtype: str

As you can see It resulted with quite a balanced classification (according to the product, that has mainly good reviews).

I then joined all the titles and the bodies in order to have a "full review" that will be helpful in few lines:

```python
df['Full Review'] = df.apply(lambda row: row['Title'] + ' ' + row['Body'], axis=1)
```

Sample of the final dataset:

<img width="1088" alt="Screenshot 2023-06-09 alle 17 08 06" src="https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/20b1fedc-0829-441a-a2c8-817fead14114">

Now, the last part of the sentiment analysis is to create a proper dataframe to feed our models. Since the standard one with all the words could not be admissible since Machine Learning Models doesn't know how to interpret a certain word in order to classify a phrase, I needed to map every review according to the word it has in it.

To do so there is a particular module of ```sklearn.feature_extraction.text```, ```TfidfVectorizer ``` that allows us to create a dense matrix in order to see for every word a level of beloging to every review:

```python
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['Full Review'])
feature_names = vectorizer.get_feature_names_out()
matrix = vectors.todense()
list_dense = matrix.tolist()
sparse_matrix = pd.DataFrame(list_dense, columns=feature_names)

sparse_matrix['Polarity_Text'] = list_polarity
```
The matrix is created starting from the column "Full Review" I created above.

Sparse Matrix:

![Screenshot 2023-06-11 alle 20 08 33](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/b2481829-abb9-405e-b37a-15673cef7337)

It is difficult to show the complete result, since, as you can see the matrix has 2603 columns (all the words of all the different reviews) and 500 rows (all the different reviews).
The different elements inside the matrix are a sort of weight every word has inside that specific review.

The weight is calculated as: 

$$
W_{x,y} = tf_{x,y} * \log{(\frac{N}{df_x})}
$$

Finally, I stored the final dataset with all the reviews and the sparse amtrix into two different CSV files:

```python
df.to_csv('data_final.csv', index=False)

sparse_matrix.to_csv('matrix.csv', index=False)
```

## CODE - PART 3: models.ipynb
For the last part of the code I applied some classifications models to my dataset. 

Firstly, I divided the dataset (the matrix) into X and Y, the Y is the target (the class Bad, Neutral or Good) while the X is the set of all the words. Then I created **train** and **test** subset of the X and Y. I choose as a size of test set 0.2 (20%) since I think it is more important to have a more robust train set since the test is used only to assess the train phase.

```python
X = sparse_matrix.iloc[:,0:-1]
Y = sparse_matrix.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

The models that I choose are:
1. Näive Bayes Classiers

```python
# Naive Bayes
nb = GaussianNB()
                                                    
nb.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_nb = accuracy_score(Y_test, y_pred)
recall_nb = recall_score(Y_test, y_pred, average='micro')
f1_score_nb = f1_score(Y_test, y_pred, average='micro')
```
> Accuracy: 0.53922<br>Recall: 0.5392156862745098<br>F1 Score: 0.5392156862745098

This one is the worst in terms of performance, given the semplicistics assumption under the model. Given that I decided to compute the confusion matrix in order to have a better comprehension of the classification the model made:

![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/edc7b50c-241d-4a3d-80f1-06e66621d11c)

2. Multinomial Bayes Classifiers

```python
fb = MultinomialNB()

fb.fit(X_train, Y_train)

# Step 5: Make predictions on the test set
y_pred = fb.predict(X_test)

# Step 6: Evaluate the performance of the classifier
accuracy_fb = accuracy_score(Y_test, y_pred)
recall_fb = recall_score(Y_test, y_pred, average='micro')
f1_score_fb = f1_score(Y_test, y_pred, average='micro')
```
> Accuracy: 0.7549<br>Recall: 0.7549019607843137<br>F1 Score: 0.7549019607843137

I thought could be interesting to compare the Näive Bayes with the Multinomial Bayes since the second works quite well with occurences of words within a text. Even though I don't have a dataset with occurences but with weights we can clearly see that the performance increased.

3. Decision Tree

```python
# Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42, splitter='best')

# Train the classifier
dt.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = dt.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_dt = accuracy_score(Y_test, y_pred)
recall_dt = recall_score(Y_test, y_pred, average='micro')
f1_score_dt = f1_score(Y_test, y_pred, average='micro')
```
> Accuracy: 0.88235<br>Recall: 0.8823529411764706<br>F1 Score: 0.8823529411764706

I used the Entropy method to split and the split decision as "best" in order to select the split with less entropy.

4. Random Forest

```python
ensemble_clfs = [
    ("RandomForestClassifier", RandomForestClassifier(warm_start=True, oob_score=True, random_state=42))
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


min_estimators = 50
max_estimators = 500

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 10):
        clf.set_params(n_estimators=i)
        clf.fit(X, Y)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))


for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err) # unzip the tuple with number of trees and OOB error
    plt.figure(figsize=(15,8))
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
```

Implementing a Random Forest I thought It could be useful to check how many trees are the best choice to construct the Forest. In the code above I created a list of tuples containing "RandomForest Classifiers" string and as second element a tuple with the number of trees used and the OOB rate. Setting the attribute ```oob_score=True``` I can track every time the OOB score that is the probability of predicting the Out Of Bag samples correctly. The last part of the code plot a line chart where on the x axes there is the number of trees (from 50 to 500 with a step of 10) and the y axes the OOB rate.

![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/8c9e8e5f-93b9-4229-af81-4b0bd4039359)

After done that I applied some hyperparameter tuning in order to visualize which are the best parameter to obtain the highest accuracy, given the fact that the Random Forest has many parameter to set:

```python
params = {'n_estimators' : [110, 350, 450],
          'criterion' : ['gini', 'entropy'],
          'max_depth' : [None, 4, 10, 15],
          'max_features' : ['sqrt', 'log2', None],
          'bootstrap' : [True, False]}

hyperparameter_tuning = GridSearchCV(RandomForestClassifier(), params, verbose=1, cv=3, n_jobs=-1)

hyp_res = hyperparameter_tuning.fit(X_train, Y_train)

hyp_res.best_params_
```
> {'bootstrap': True,<br>'criterion': 'gini',<br>'max_depth': 10,<br>'max_features': None,<br>'n_estimators': 110}
 
 Then I set as parameters the one I got from the tuning:

```python
# Random Forest
rf = RandomForestClassifier(n_estimators=110, criterion='gini', random_state=42, bootstrap=True, max_depth=10, max_features=None)

# Train the classifier
rf.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_rf = accuracy_score(Y_test, y_pred)
recall_rf = recall_score(Y_test, y_pred, average='micro')
f1_score_rf = f1_score(Y_test, y_pred, average='micro')
```
> Accuracy: 0.92157<br>Recall: 0.9215686274509803<br>F1 Score: 0.9215686274509803

As I did for the worst, I computed the confusion matrix also for the best model:

![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/5178c774-628c-4c04-84a6-510796517bcf)

5. K-Neigherest Neighbors

Since finding the optimal value of k cuould be a trivial task I decided to compute the accuracy according to different level of k:

```python
mean_acc_knn = []

for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    mean_acc_knn.append(accuracy_score(Y_test, y_pred))

loc = np.arange(1,21,step=1)
plt.figure(figsize = (15, 8))
plt.plot(range(1,21), mean_acc_knn)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()
```

![image](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/a4beff24-b6d8-49aa-89dc-fca8223f81b7)

As you can see from the plot the best value seems to be 5.

Then I decided to try the hyperparameter tuning also here, even though the number of parameters are different:
```python
params = {'n_neighbors' : [3, 5, 9], 
          'weights' : ['uniform', 'distance'],
          'metric' : ['minkowski', 'euclidean', 'manhattan']}

hyperparameter_tuning = GridSearchCV(KNeighborsClassifier(), params, verbose=1, cv=3, n_jobs=-1)

hyp_res = hyperparameter_tuning.fit(X_train, Y_train)

hyp_res.best_params_
```
> {'metric': 'minkowski', 'n_neighbors': 9, 'weights': 'distance'}

```python
# Create a KNN classifier
k = 9  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance')

# Train the classifier
knn.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_knn = accuracy_score(Y_test, y_pred)
recall_knn = recall_score(Y_test, y_pred, average='micro')
f1_score_knn = f1_score(Y_test, y_pred, average='micro')
print("Accuracy:", round(accuracy_knn, 5))
print("Recall:", recall_knn)
print("F1 Score:", f1_score_knn)
```
> Accuracy: 0.80392<br>Recall: 0.803921568627451<br>F1 Score: 0.8039215686274509

The last model I implemented is the Logistic Regression:

```pyhton
# Initialize the Logistic Regression model
logreg = LogisticRegression(random_state=42)

# Fit the model to the training data
logreg.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the model
accuracy_lr = accuracy_score(Y_test, y_pred)
recall_lr = recall_score(Y_test, y_pred, average='micro')
f1_score_lr = f1_score(Y_test, y_pred, average='micro')
print("Accuracy:", accuracy_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_score_lr)
```
> Accuracy: 0.7549019607843137<br>Recall: 0.7549019607843137<br>F1 Score: 0.7549019607843137

This is basically the end of the analysis. The last thing I did has been to store all the result in a dataFrame:

```python
results = pd.DataFrame({'Model Name' : ['Naive Bayes', 'Full Bayes' ,'Decision Tree', 'Random Forest', 'KNN'], 
                                'Accuracy' : [accuracy_nb, accuracy_fb , accuracy_dt, accuracy_rf, accuracy_knn],
                                'Recall' : [recall_nb, recall_fb, recall_dt, recall_rf, recall_knn],
                                'F1 Score' : [f1_score_nb, f1_score_fb, f1_score_dt, f1_score_rf, f1_score_knn]})
```

![Screenshot 2023-06-10 alle 16 01 32](https://github.com/riccardo-borgo/Sentiment-Analysis_Amazon/assets/51230348/7e28e710-ac7b-4659-ad2c-e4fb1a4cf038)

As you can see the most powerful model is the Random Forest, probably because it is also the most complex one and is able to grasp more information from the matrix respect to other models. 











