"""
classify.py
"""
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd
# from sklearn.cross_validation import KFold
import numpy as np
# from sklearn.metrics import accuracy_score

labeled_tweets = []
def remove_duplicate_tweets(tweets):
    filtered_tweets = set()
    for tweet in tweets:
        if tweet not in filtered_tweets:
            filtered_tweets.add(tweet)

    filtered_list = list(filtered_tweets)
    print(len(filtered_list))
    return filtered_list

# def add_labels_to_tweets(tweets):
#     # filtered_tweets = remove_duplicate_tweets(tweets)
#     affinitySet = {1,0,-1}
#     print("Please lable tweets as 1: positive, -1: negative and 0: neutral")
#     for i in range(len(tweets)):
#         print(tweets[i])
#         ip = input()
#         ip = int(ip)
#         labeled_tweets.append((tweets[i],ip))
#     return labeled_tweets

def add_labels_to_tweets(tweets):
    # filtered_tweets = remove_duplicate_tweets(tweets)
    affinitySet = {1,0,-1}
    labeled_tweets = []
    print("Please lable tweets as 1: positive, -1: negative and 0: neutral")
    for i in range(len(tweets)):
        correctInput = False
        while not correctInput:
            print(tweets[i])
            ip = input()
            try:
                ip = int(ip)
            except:
                pass
            if(ip in affinitySet):
                labeled_tweets.append((tweets[i],ip))
                correctInput = True
                break
            else:
                print("Please Enter the correct value 1: positive, -1: negative and 0: neutral")
                correctInput = False
    return labeled_tweets

def get_affinity_tweets(data, positive = 1, negative=-1, neutral =0):
    positive_list = []
    negative_list =[]
    neutral_list = []
    for k, v in data:
        value = int(v)
        if value == positive:
            positive_list.append((k,value))
        if value == negative:
            negative_list.append((k,value))
        if value == neutral:
            neutral_list.append((k,value))
    return positive_list,neutral_list, negative_list

# def find_affinity_for_affin_data(data, pos=4, neg=0, neu=2):
#     positive_list = []
#     negative_list = []
#     neutral_list = []
#     for k, v in data:
#         value = int(v)
#         if value == pos:
#             positive_list.append((k, value))
#         if value == neg:
#             negative_list.append((k, value))
#         if value == neu:
#             neutral_list.append((k, value))
#     return positive_list, neutral_list, negative_list

def tokenize(data):
    return_val = []
    values = re.split(r'\s+',re.sub(r'[?|$|.|\d|!]',r'', data.strip()))
    for val in values:
        if len(val) >= 3 and "@" not in val and "#" not in val and "https" not in val:
            return_val.append(val)
    return return_val

def get_words(tweets):
    words_list = []
    for k, v in tweets:
        tweet = tokenize(k)
        words_list.append((tweet,v))
    return words_list

def get_word_count(word_list):
    list_val = {}
    for iter in range(len(word_list)):
        elements = word_list[iter][0]
        print(elements)
        for ele in elements:
            if ele in list_val:
                list_val[ele] +=1
    return list_val

# def cross_validation_accuracy(clf, X, labels, k):
#     cv = KFold(n_splits=k)
#     accuracies = []
#     for train_idx, test_idx in cv.split(X):
#         clf.fit(X[train_idx], labels[train_idx])
#         predicted = clf.predict(X[test_idx])
#         accuracies.append(accuracy_score(labels[test_idx], predicted))
#     avg = np.mean(accuracies)
#     return avg

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(n_splits=k)
    accuracies = []
    labels = np.array(labels)
    for train_idx, test_idx in cv.split(X):
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        accuracies.append(accuracy_score(labels[test_idx], predicted))
    avg = np.mean(accuracies)
    return avg

def train_test_and_predict(tweet_collected_test_Data,tweets_train_data, affin_data=False,KFold=25, maximum_df=0.8,minimum_df=5):
    train_data = []
    test_data =[]
    label_train = []
    for tweet in tweets_train_data:
        train_data.append(tweet[0])
        label_train.append(tweet[1])
    for t in tweet_collected_test_Data:
        test_data.append(t)
    if not affin_data:
        vectorize = CountVectorizer(input=tweet_collected_test_Data,tokenizer=tokenize, ngram_range=(1,1),max_df=maximum_df,min_df=minimum_df,binary=True, stop_words=None)
        # vectorize = TfidfVectorizer(input="tweet_collected", tokenizer=tokenize, ngram_range=(1, 1), max_df=maximum_df, stop_words=None, min_df=minimum_df, binary=True, sublinear_tf=True, use_idf=True)
    else:
        vectorize = TfidfVectorizer(min_df=minimum_df, max_df=maximum_df, sublinear_tf=True, use_idf=True)

    vTrain = vectorize.fit_transform(train_data)
    vTest = vectorize.transform(test_data)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(vTrain,label_train)
    score = cross_validation_accuracy(classifier,vTrain, label_train, KFold)
    print("K cross Validation accuracy is: ", score)
    prediction = classifier.predict(vTest)
    return prediction

def get_affinity_count(values, positive = 1, negative = -1, neutral = 0):
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    for val in values:
        if val == positive:
            positive_count += 1
        elif val == neutral:
            neutral_count += 1
        elif val == negative:
            negative_count += 1
    return positive_count,neutral_count,negative_count

def get_labeled_data(data, labelDataFileName):
    try:
        labeled_data = pickle.load(open(labelDataFileName, 'rb'))
    except:
        print("Training first 80 tweets from all tweets")
        data_encoded = safe_stringify(data)
        labeled_data = add_labels_to_tweets(data_encoded)
        pickle.dump(labeled_data, open(labelDataFileName, 'wb'))
    return labeled_data

def write_data(pos_count, neg_count, neutral_count, positive_tweets, negative_tweets, neutral_tweets, fileName):
    classified_data = {}
    classified_data['positive_count'] = pos_count
    classified_data['negative_count'] = neg_count
    classified_data['neutral_count'] = neutral_count
    classified_data['positive_tweets'] = positive_tweets[1][0]
    classified_data['negative_tweets'] = negative_tweets[1][0]
    classified_data['neutral_tweets'] = neutral_tweets[1][0]
    pickle.dump(classified_data, open(fileName, 'wb'))

def safe_stringify(data):
    return_data = []
    for i in data:
        return_data.append(str(i.encode("ascii","ignore"))[2:-1])
    return return_data

def safe_stringify_for_tuple(data):
    return_data = []
    for k,v in data:
        return_data.append(((str(k.encode("ascii", "ignore"))[2:-1]),v))
    return return_data

def main():
    print("********* Classification Phase **********")
    collectedDataFileName = 'collected_data.p'
    labelDataFileName = 'labeled_data.p'
    affinDataFileName = 'train-data.csv'

    data = pickle.load(open(collectedDataFileName,'rb'))
    all_tweets = safe_stringify(data['Tweets'])
    print("Number of Tweets:")
    print(len(all_tweets))
    labeled_data = get_labeled_data(all_tweets[:80], labelDataFileName)
    labeled_data = safe_stringify_for_tuple(labeled_data)
    tweet_collected = all_tweets[80:100]
    positive_list, neutral_list, negative_list = get_affinity_tweets(labeled_data)
    tweets = positive_list+neutral_list+negative_list
    print("Test train and predict for manually trained data taken only 20 from first 100 tweets")
    predict_values = train_test_and_predict(tweet_collected,tweets)
    print("Predicted values:")
    print(predict_values[:200])
    pos_count, neu_count,neg_count = get_affinity_count(predict_values)
    write_data(pos_count,neg_count,neu_count,positive_list,negative_list,neutral_list,fileName='classified_data.p')

    print("***********************")
    print("Test for Affin data")
    print("Values for first 200 predictions of affin data")
    train_tweet_data = pd.read_csv(affinDataFileName,encoding='utf-8')
    polarity = train_tweet_data['Polarity']
    tweet_data = train_tweet_data['Tweet']
    for i in range(len(polarity)):
        labeled_tweets.append((tweet_data[i],polarity[i]))
    positive_list, neutral_list, negative_list = get_affinity_tweets(labeled_tweets,positive=4,negative=0,neutral=2)
    affin_tweets = positive_list+neutral_list+negative_list
    affin_tweets = safe_stringify_for_tuple(affin_tweets)
    predict_values = train_test_and_predict(all_tweets,affin_tweets,affin_data=True,KFold=35)
    print("Predicted values:")
    print(predict_values[:200])
    pos_count, neu_count, neg_count = get_affinity_count(predict_values, positive=4,negative = 0, neutral = 2)
    write_data(pos_count, neg_count, neu_count, positive_list, negative_list, neutral_list, 'classified_affin_data.p')
    # classfied_affin_data = {}
    # classfied_affin_data['positive_count']= pos_count
    # classfied_affin_data['negative_count'] = neg_count
    # classfied_affin_data['neutral_count'] = neu_count
    # classfied_affin_data['positive_tweets'] = positive_list[0][0]
    # classfied_affin_data['negative_tweets'] = negative_list[1][0]
    # classfied_affin_data['neutral_tweets'] = neutral_list[1][1]
    # pickle.dump(classfied_affin_data,open('classified_affin_data.p','wb'))

if __name__ == '__main__':
    main()