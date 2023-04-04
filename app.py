from flask import Flask, render_template, request



import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



# Load data
news_df = pd.read_csv('./Notebook/Data/train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/newspaper', methods=['POST'])
def newspaper():
    if request.method == 'POST':
        news = request.form['news']
        # print(type(news))
        # print(news)
        news_data=vector.transform([news])
        prediction = model.predict(news_data)
        if prediction[0] == '0':
            message = 'The News You Have Entered Is Real'
            color ='green'
            parameter=0
        else:
             message = 'The News You Have Entered Is Fake'
             color ='red'
             parameter=1
        return render_template('newspaper.html', message=message,color=color,parameter=parameter)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
