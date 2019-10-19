import re
import nltk
import string
import enchant
import operator
from nltk.corpus import stopwords
from collections import OrderedDict
from textblob import TextBlob, Word
from textblob import Blobber
from textblob.taggers import NLTKTagger
import pandas as pd

url = "http://m4y4nk.net/hackabit/canon.txt"
df = pd.read_csv(url, error_bad_lines=False)
apostropheList = {"n't" : "not","aren't" : "are not","can't" : "cannot","couldn't" : "could not","didn't" : "did not","doesn't" : "does not", \
				  "don't" : "do not","hadn't" : "had not","hasn't" : "has not","haven't" : "have not","he'd" : "he had","he'll" : "he will", \
				  "he's" : "he is","I'd" : "I had","I'll" : "I will","I'm" : "I am","I've" : "I have","isn't" : "is not","it's" : \
				  "it is","let's" : "let us","mustn't" : "must not","shan't" : "shall not","she'd" : "she had","she'll" : "she will", \
				  "she's" : "she is", "shouldn't" : "should not","that's" : "that is","there's" : "there is","they'd" : "they had", \
				  "they'll" : "they will", "they're" : "they are","they've" : "they have","we'd" : "we had","we're" : "we are","we've" : "we have", \
				  "weren't" : "were not", "what'll" : "what will","what're" : "what are","what's" : "what is","what've" : "what have", \
				  "where's" : "where is","who'd" : "who had", "who'll" : "who will","who're" : "who are","who's" : "who is","who've" : "who have", \
				  "won't" : "will not","wouldn't" : "would not", "you'd" : "you had","you'll" : "you will","you're" : "you are","you've" : "you have"}

stopWords = stopwords.words("english")
exclude = set(string.punctuation)
exclude.remove("_")
linkPtrn = re.compile("^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$")

# English vocabulary
enchVocab = enchant.Dict("en_US")
vocabList = set(w.lower() for w in nltk.corpus.words.words())
maxHops = 4

def fileCreation(reviewContent, filename):
    phrasesDict = dict()

    for a in range(len(reviewContent)):  # Stores the score of the nouns
        for i in range(len(reviewContent[a])):
            line_words = reviewContent[a][i]

            phrases = TextBlob(line_words).noun_phrases
            for p in phrases:
                if (len(p.split()) == 2):
                    if (p not in phrasesDict):
                        phrasesDict[p] = 1
                    else:
                        phrasesDict[p] += 1
    filterAdj1(phrasesDict, filename)


def filterAdj1(phrasesDict, filename):
    phrasesDict = OrderedDict(sorted(phrasesDict.items(), key=operator.itemgetter(1), reverse=True))
    newPhrases = dict()
    exclude = set(string.punctuation)
    exclude.remove("_")
    for line_words, count in phrasesDict.items():
        # Preprocessing text
        line_words = ' '.join([apostropheList[word] if word in apostropheList else word for word in line_words.split()])
        line_words = ''.join(ch for ch in line_words if ch not in exclude)
        line_words = re.sub(r' [a-z][$]? ', ' ', line_words)
        line_words = [Word(word).lemmatize() for word in line_words.split() if
                      (word not in stopwords.words("english") and not word.isdigit()) and len(word) > 2]
        line_words = ' '.join(line_words)
        if (len(line_words.strip(" ").split()) == 2):
            if (line_words in newPhrases):
                newPhrases[line_words] += count
            else:
                newPhrases[line_words] = count
    # Bigrams from the file
    newPhrases = OrderedDict(sorted(newPhrases.items(), key=operator.itemgetter(1), reverse=True))

    # Applying Threshold to Bigrams
    nouns1 = []
    for key, value in newPhrases.items():
        if value >= 3:
            nouns1.append(key)

    stopWords = stopwords.words("english")
    exclude = set(string.punctuation)
    reviewContent = []

    with open(filename) as f:
        for line in f:
            l1 = line[1:-3]
            l1 = l1.strip()
            reviewContent.append(l1.rstrip("\r\n"))

        # tb = Blobber(pos_tagger=PerceptronTagger())
        tb = Blobber(pos_tagger=NLTKTagger())
        nounScores = dict()

        # Writing to a file
        f = open('modified.txt', 'w')
        for a in range(len(reviewContent)):
            # Finding bigrams in review
            for i in range(len(reviewContent[a])):
                text = reviewContent[a][i]
                x = tb(text).tags  # NLTK tagger

                tagList = []
                e = 0
                f.write("##")

                while e < len(x):
                    tagList = ""
                    temp = ""
                    wrt = x[e][0]
                    e = e + 1
                    count = e
                    tp = 0
                    if (count < len(x) and (x[count - 1][1] == "NN" or "JJ") and (x[count][1] == "NN" or "JJ")):
                        tagList = x[count - 1][0] + " " + x[count][0]
                        temp = x[count][0]
                        count = count + 1
                    if tagList != "":
                        # Checking if consecutive nouns we found out are in noun phrases
                        if tagList in nouns1:
                            tagList = tagList.replace(' ', '')
                            f.write(tagList)
                            tp = 1
                            e = count
                    if tp == 0:
                        f.write(wrt)
                    f.write(" ")
                f.write(".\r\n")


def findFeatures(reviewContent,filename):
    nounScores = dict()

    adjDict = dict()
    tb = Blobber(pos_tagger=NLTKTagger())

    for a in range(len(reviewContent)):  # Stores the score of the nouns
        #print("printing words::::")
        #print(reviewContent[a])
        text = ' '.join([word for word in reviewContent[a].split() if word not in stopwords.words("english")])
        text = ''.join(ch for ch in text if ch not in exclude)
        text = nltk.word_tokenize(text)
        x = nltk.pos_tag(text)

        # Get the noun/adjective words and store it in tagList
        tagList = []
        for e in x:
            if (e[1] == "NN" or e[1] == "JJ"):
                tagList.append(e)

        # Add the nouns(which are not in the nounScores dict) to the dict
        for e in tagList:
            if e[1] == "NN":
                if e[0] not in nounScores:
                    nounScores[e[0]] = 0

        # For every adjective, find nearby noun
        for l in range(len(tagList)):
            if ("JJ" in tagList[l][1]):
                j = k = leftHop = rightHop = -1

                for j in range(l + 1, len(tagList)):
                    if (j == l + maxHops):
                        break
                    if ("NN" in tagList[j][1]):
                        rightHop = (j - l)
                        break

                for k in range(l - 1, -1, -1):
                    if (j == l - maxHops):
                        break
                    if ("NN" in tagList[k][1]):
                        leftHop = (l - k)
                        break

                # Compare which noun is closer to adjective(left or right) and assign the adj to corresponding noun
                if (leftHop > 0 and rightHop > 0):
                    if (leftHop - rightHop) >= 0:
                        adjDict[tagList[l][0]] = tagList[j][0]
                        nounScores[tagList[j][0]] += 1
                    else:
                        adjDict[tagList[l][0]] = tagList[k][0]
                        nounScores[tagList[k][0]] += 1
                elif leftHop > 0:
                    adjDict[tagList[l][0]] = tagList[k][0]
                    nounScores[tagList[k][0]] += 1
                elif rightHop > 0:
                    adjDict[tagList[l][0]] = tagList[j][0]
                    nounScores[tagList[j][0]] += 1

    nounScores = OrderedDict(sorted(nounScores.items(), key=operator.itemgetter(1)))
    return filterAdj(nounScores, adjDict, filename)


def filterAdj(nounScores, adjDict, filename):
    adjectList = list(adjDict.keys())
    nouns = []
    for key, value in nounScores.items():
        if value >= 3:
            nouns.append(key)
    nouns = set(nouns)

    stopWords = stopwords.words("english")
    exclude = set(string.punctuation)
    reviewContent = []

    with open(filename) as f:
        review = []
        for line in f:
            l1 = line[1:-3]
            l1 = l1.strip()
            reviewContent.append(l1.rstrip("\r\n"))

    tb = Blobber(pos_tagger=NLTKTagger())
    nounScores = dict()
    f = open('modified.txt', 'w')
    for a in range(len(reviewContent)):
        f.write("\r\n")
        text = reviewContent[a]
        x = tb(text).tags
        tagList = []
        e = 0
        f.write("##")

        while e < len(x):
            tagList = []
            f.write(x[e][0])
            e = e + 1
            count = e
            if (count < len(x) and x[count - 1][1] == "NN" and x[count][1] == "NN"):
                tagList.append(x[count - 1][0])

                while (count < len(x) and x[count][1] == "NN"):
                    tagList.append(x[count][0])
                    count = count + 1
            if tagList != [] and len(tagList) == 2:
                if set(tagList) <= nouns:

                    for t in range(1, len(tagList)):
                        f.write(tagList[t])
                    e = count
            f.write(" ")
        f.write(".\r\n")

    return adjectList
