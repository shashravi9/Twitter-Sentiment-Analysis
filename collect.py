"""
collect.py
"""
from TwitterAPI import TwitterAPI
import sys
import time
import configparser
import urllib.parse
import random as rand
import pickle

# def get_twitter(config_file):
#     """ Read the config_file and construct an instance of TwitterAPI.
#     Args:
#       config_file ... A config file in ConfigParser format with Twitter credentials
#     Returns:
#       An instance of TwitterAPI.
#     """
#     config = configparser.ConfigParser()
#     config.read(config_file)
#     twitter = TwitterAPI(
#                    config.get('twitter', 'consumer_key'),
#                    config.get('twitter', 'consumer_secret'),
#                    config.get('twitter', 'access_token'),
#                    config.get('twitter', 'access_token_secret'))
#     return twitter

consumer_key = 'ykmeCgBSGVolLuW6mZDmzQYMN'
consumer_secret = 'WbsjJRiu9t74Er6PtTAvjPi06uIUima0bdEBvG0wHM8aW5gkLu'
access_token = '780274970215809024-79YZpAqS725vsFst8fEb96397jI1FrA'
access_token_secret = 'LwlUpHDuAqz6NGXRczrXmL8eN5Hmu1GJJSnAonMHLolY5'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_tweets(twitter,hashtag, isSleep=False, sleepTime_min = 1):
    tweets_F = []
    statuses_F = []
    user_id_list = []
    screen_name_list = []
    for i in range(20):
        tweets = robust_request(twitter, 'search/tweets',
                                {'q': hashtag + ' -filter:retweets', 'count': 100, 'lang': 'en','result_type': 'mixed'}).json()
        statuses = tweets['statuses']

        for i in range(len(statuses)):
            if statuses[i]['retweeted'] == False:
                tweets_F.append(statuses[i]['text'])
                statuses_F.append(statuses[i])
        if(isSleep):
            print("Sleeping for %d minutes" %sleepTime_min)
            time.sleep(61*sleepTime_min)

    for iter in range(10):
        index = rand.randrange(len(statuses_F)- 1)
        userId = statuses_F[index]['user']['id']
        screenName = statuses_F[index]['user']['screen_name']
        if userId not in user_id_list:
            user_id_list.append(userId)
        if screenName not in screen_name_list:
            screen_name_list.append(screenName)

    return tweets_F, user_id_list, screen_name_list

def get_friends(twitter, screenNameList):
    friend_dict = {}
    for screen_name in screenNameList:
        result = robust_request(twitter, 'friends/ids', {'screen_name': screen_name, 'count': 100}).json()
        friend_dict[screen_name] = result['ids']
    return friend_dict

def get_followers(twitter, screenNameList):
    follower_dict = {}
    for screen_name in screenNameList:
        result = robust_request(twitter,'followers/ids',{'screen_name':screen_name,'count': 100}).json()
        follower_dict[screen_name] = result['ids']
    return follower_dict


def main():
    print("***********Collection phase*********")
    twitter = get_twitter()
    hashtag = 'JusticeLeague'
    collectDataFileName = 'collected_data.p'
    tweets_filtered,userIds, screenNames = get_tweets(twitter,hashtag)
    friends_ids = get_friends(twitter,screenNames)
    followers_ids = get_followers(twitter,screenNames)
    print("Number of Tweets collected: %d" %len(tweets_filtered))
    print("User Ids is",userIds)
    print("Screen names is", screenNames)
    data = {'UserIds': userIds, 'ScreenNames': screenNames, 'Tweets': tweets_filtered, 'Friend_Ids': friends_ids,
            'Follower_Ids': followers_ids}
    pickle.dump(data,open(collectDataFileName,'wb'))

if __name__ == '__main__':
    main()
