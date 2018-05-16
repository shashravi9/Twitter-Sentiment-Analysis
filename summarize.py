"""
sumarize.py
"""
import pickle

def print_data(dict):
    print("Number of instances per class found:")
    print("Positive Count: %d" % (dict['positive_count']))
    print("Negative Count: %d" % (dict['negative_count']))
    print("Neutral Count: %d" % (dict['neutral_count']))
    print("Positive Tweets: %s" % (dict['positive_tweets']))
    print("Negative Tweets: %s" % (dict['negative_tweets']))
    print("Neutral Tweets: %s" % (dict['neutral_tweets']))

def write_data(file, dict):
    file.write("Number of instances per class found:\n")
    file.write("Positive Count: %d\n" % (dict['positive_count']))
    file.write("Negative Count: %d\n" % (dict['negative_count']))
    file.write("Neutral Count: %d\n" % (dict['neutral_count']))
    file.write("Positive Tweets: %s\n" % (dict['positive_tweets']))
    file.write("Negative Tweets: %s\n" % (dict['negative_tweets']))
    file.write("Neutral Tweets: %s\n" % (dict['neutral_tweets']))
    # return file

def write_data_to_file(userIds, tweets, clustered_data, classified_data, classified_affin_data):
    count = 0
    for i in range(len(clustered_data)):
        count += len(clustered_data[i].nodes())

    file = open('summarize.txt', 'w')
    file.write("Number of users collected: %d\n" %(len(userIds)))
    file.write("Number of Tweets collected: %d\n" %(len(tweets)))
    file.write("Number of communities discovered: %d\n" %(len(clustered_data)))
    file.write("Average number of users per community: Average = %f\n" %(count/len(clustered_data)))
    file.write("********************************\n")
    file.write("Manually Trained and Tested data\n")
    file.write("********************************\n")
    write_data(file, classified_data)
    file.write("*******************************************************\n")
    file.write("For Affin train data\n")
    file.write("********************************\n")
    write_data(file,classified_affin_data)
    file.close()

def main():
    print("**********Summary phase************")
    collectedDataFileName = 'collected_data.p'
    classifiedDataFileName = 'classified_data.p'
    classifiedAffinDataFileName = 'classified_affin_data.p'
    clusteredDataFileName = 'clustered_data.p'
    data = pickle.load(open(collectedDataFileName,'rb'))
    userIds = data['UserIds']
    tweets = data['Tweets']
    classified_data = pickle.load(open(classifiedDataFileName,'rb'))
    classified_affin_data = pickle.load(open(classifiedAffinDataFileName, 'rb'))
    clustered_data = pickle.load(open(clusteredDataFileName,'rb'))

    print("Number of users collected: %d" %(len(userIds)))
    print("Number of Tweets collected: %d" %(len(tweets)))
    print("Number of communities discovered: %d" %(len(clustered_data)))
    count =0
    for i in range(len(clustered_data)):
        count+=len(clustered_data[i].nodes())
    print("Average number of users per community: Average = %f" %(count/len(clustered_data)))
    print("Manually Trained and Tested data")
    print("********************************************************")
    print_data(classified_data)
    print("*******************************************************")
    print("For Affin train data")
    print_data(classified_affin_data)
    print("Writing all these data to file")
    write_data_to_file(userIds, tweets, clustered_data, classified_data, classified_affin_data)

if __name__ == '__main__':
        main()