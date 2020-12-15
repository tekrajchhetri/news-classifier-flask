from flask import Flask, render_template, url_for, request
import  os, joblib, sklearn

#vectorizer
vectorizer = open(os.path.join("static/models/final_news_cv_vectorizer.pkl"),"rb")
news_cv = joblib.load(vectorizer)

app = Flask(__name__)

def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        text = request.form['textareatext']
        if len(text.strip()) == 0:
            final_prediction = "No content to predict"
        else:

            vectorizer_text =  news_cv.transform([text]).toarray()

            selected_model = request.form['selectmodel']

            if selected_model == "rf":
                rfmodel = open(os.path.join("static/models/newsclassifier_RFOREST_model.pkl"),"rb")
                news_clf = joblib.load(rfmodel)
            elif selected_model == "nb":
                nbmodel = open(os.path.join("static/models/newsclassifier_NB_model.pkl"),"rb")
                news_clf = joblib.load(nbmodel)
            elif  selected_model == "logit":
                lrmodel = open(os.path.join("static/models/newsclassifier_Logit_model.pkl"),"rb")
                news_clf = joblib.load(lrmodel)

            #predictions
            predictions_label = {"business":0, "tech":1, "sport":2, "health":3, "politics":4,"entertainment":5}
            predict = news_clf.predict(vectorizer_text)
            final_prediction = get_keys(predict, predictions_label)
            
        return render_template("index.html",
                               newscontent=text.upper(),
                               prediction_result=final_prediction.upper())

if __name__ == "__main__":
    app.run(debug=True)