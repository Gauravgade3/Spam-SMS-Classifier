import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request

ps = PorterStemmer()

app = Flask(__name__)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@app.route("/", methods=['GET', 'POST'])
# @cross_origin()
def home ():

    return render_template("home.html")









@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    if request.method == "POST":
        input_sms = request.form['text']
        transformed_sms = transform_text(input_sms)
    # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
    # 3. predict
        result = model.predict(vector_input)[0]
    # 4. Display
        if result == 1:
            label="Spam"
        else:
            label="Not Spam"




    return render_template('result.html', prediction_text="This SMS is {}".format(label))




if __name__ == '__main__':
    app.run(debug=True)