# switched from tflearn to tf.keras for tf.v2
import nltk
import json
import numpy
import random
import pickle
# import tflearn
import tensorflow as tf

from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense

stemmer = LancasterStemmer()
nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

# use saved pickle file or prepare the data again
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp = nltk.word_tokenize(pattern)
            words.extend(temp)
            docs_x.append(temp)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        temp = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in temp:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# create the model
# tf.compat.v1.reset_default_graph()

# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)
# model = tflearn.DNN(net)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=len(training[0])))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(len(output[0]), activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# save & load the model
try:
    tf.keras.models.load_model("./model_temp")
    # model.load("model.tflearn")
except:
    model.fit(training, output, epochs=1000, batch_size=8)
    tf.keras.models.save_model(model, "./model_temp")
    # model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
            
        results = model.predict([bag_of_words(inp, words)])
        # confidence value
        resutls_index = numpy.argmax(results)
        tag = labels[resutls_index]

        # compare confidence value with a pre-setted threshold
        if results[resutls_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["response"]
        
            print(random.choice(responses))
        else:
            print("didn't get that")

chat()