from importlib.resources import path
import json
import pickle
import math
import os

from sklearn import preprocessing
from models import *
from models.models import CNNModel, GRUModel, LSTMModel
from preprocessing import *
from pyexpat import model
from flask import request, abort, make_response, redirect, render_template
from flask import Flask
from flask import jsonify
from json import dumps
from scrapping import *
from catboost import CatBoostClassifier
import torch
import os
from numpy import np
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(topic_model, topic, save:bool = True):
    plt.figure(figsize=(15,15))
    text = {word: value for word, value in topic_model.get_topic(topic)}
    wc = WordCloud(background_color="black", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join('./static/', 'wordscloud.png'))



app = Flask(__name__)

@app.before_first_request
def load_global_data():
    global model
    
    with open(os.path.join('models/', 'knn_classifier.pkl'), 'rb') as fid:
        model = pickle.load(fid)


def get_prediction(loader):
    WEIGHTS = np.array([1, 1.2])
    
    cnn_logits = CNNModel_(loader).detach().numpy()
#     resnet152_logits = resnet152(transformed_image[None,]).detach().numpy()
    gru_logits = GRUModel_(loader).detach().numpy()
    
    predictions = np.array([0, 0])
    
    boosting_cnn_probas = boosting.predict_proba(gru_logits)
    boosting_gru_probas = boosting.predict_proba(cnn_logits)
    
    forest_cnn_probas = random_forest.predict_proba(gru_logits)
    forest_gru_probas = random_forest.predict_proba(cnn_logits)
    
    svm_cnn_probas = svm.predict_proba(gru_logits)
    svm_gru_probas = svm.predict_proba(cnn_logits)
    
    predictions = (boosting_cnn_probas + boosting_gru_probas + 
                forest_cnn_probas + forest_gru_probas + 
                svm_cnn_probas + svm_gru_probas) / 4
    predictions = predictions * WEIGHTS
    predictions /= predictions.sum()
    return predictions



def bad_request(message, code=400):
    abort(make_response(jsonify(message=message), code))

@app.route('/analyse', methods=['GET'])
def analyze():
    MODEL_FOLDER = './models/stacking/'
    url = request.args.get('link', type=str)
    boosting = CatBoostClassifier().load_model(os.path.join(MODEL_FOLDER, 'boosting.model'))
    random_forest = pickle.load(open(os.path.join(MODEL_FOLDER, 'random_forest.pkl'), 'rb'))
    svm = pickle.load(open(os.path.join(MODEL_FOLDER, 'svm.pkl'), 'rb'))
    preprocessor = Preprocessor(nltk=True)
    CNNModel_ = CNNModel()
    CNNModel_.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'cnn.pt')))
    CNNModel_.eval();

    LSTMModel_ = LSTMModel()
    LSTMModel_.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'lstm.pt')))
    LSTMModel_.eval();

    GRUModel_ = GRUModel()
    GRUModel_.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'gru.pt')))
    GRUModel_.eval();

    boosting = CatBoostClassifier().load_model(os.path.join(MODEL_FOLDER, 'boosting.model'))
    random_forest = pickle.load(open(os.path.join(MODEL_FOLDER, 'random_forest.pkl'), 'rb'))
    svm = pickle.load(open(os.path.join(MODEL_FOLDER, 'svm.pkl'), 'rb'))
    scrapper = YouTubeScrapper()
    API_scrapper = YouTubeAPIscrapper()
    video_info = scrapper.scrape_comments(url)
    comments = API_scrapper.get_comments(url)
    loader = preprocessor.preprocess(comments)
    most_likeble = comments.sort_values(by='likes').iloc[0,0]
    ml_comment_color = 'red' if get_prediction(most_likeble) == 0 else 1
    ml_comment_sentences = most_likeble.split('.')
    ml_comment_sentence_colors = [get_prediction(sentence) for sentence in ml_comment_sentences]
    ml_comment_json = json.dumps({
            'color': ml_comment_color, 
            'sentences': ml_comment_sentences, 
            'sentence_colors': ml_comment_sentence_colors})
    most_comment = comments.sort_values(by='comments').iloc[0,0]

    mc_comment_color = 'green' if get_prediction(most_commented) == 1 else 0
    mc_comment_sentences = most_comment.split('.')
    mc_comment_sentence_colors = [get_prediction(sentence) for sentence in ml_comment_sentences]
    mc_comment_json = json.dumps({
            'color': mc_comment_color, 
            'sentences': mc_comment_sentences, 
            'sentence_colors': mc_comment_sentence_colors})
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(comments)

    create_wordcloud(topic_model, topic=1)
    youtube_title = video_info['label'].values()[0]
    path_word_clouds = os.path.join('./static/', 'wordscloud.png')
    probas = get_prediction(loader)
    predicted_class = [np.argmax(probas_) for probas_ in probas]
    count_positive = sum(predicted_class)
    count_negative = len(predicted_class) - count_positive
    positive_percentage = count_positive // len(predicted_class)
    negative_percentage = negative_percentage // len(predicted_class)
    ml_value_of_likes = most_likeble.head(1)['likes'].values()[0]
    ml_value_of_comments = most_likeble.head(1)['comments'].values()[0]
    mc_value_of_likes = most_comment.head(1)['likes'].values()[0]
    mc_value_of_comments = most_comment.head(1)['comments'].values()[0]
    return render_template('analysis.html', 
                            youtube_title=youtube_title,
                            path_word_clouds=path_word_clouds,
                            positive_percentage=positive_percentage,
                            negative_percentage=negative_percentage,
                            context_str=url,

                            ml_comment_json=ml_comment_json,
                            mc_comment_json=mc_comment_json,

                            ml_value_of_likes=ml_value_of_likes,
                            ml_value_of_comments=ml_value_of_comments,
                            mc_value_of_likes=mc_value_of_likes,
                            mc_value_of_comments=mc_value_of_comments
                            )

@app.route('/')
def main_page():
    return render_template('main_page.html')
    
if __name__ == '__main__':
    app.run()