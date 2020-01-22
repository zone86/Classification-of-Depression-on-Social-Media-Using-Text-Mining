from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


consumer_key = 'tQWU3JxHrZOp90nkVBeZP9G7j'
consumer_secret = '7EELpz733yOdrIMY0irKQxEVMRskeMy5rtcvdFKc3zwGTK7ZZB'
access_token = '2534606803-fCBxMJHz8FKYVPjUfCIWh0FNMWbEi4PnOEgcx00'
access_secret = 'iKOODx6esFQIPDjRV9quVgWbllc2MKieni5wgD16QhFlz'



class StdOutListener(StreamListener):

    def on_data(self, data):
        with open('data/tweetdata.txt','a') as tf:
            tf.write(data)
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':


    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    stream = Stream(auth, l)

    stream.filter(track=['depression', 'anxiety', 'mental health', 'suicide', 'stress', 'sad'])