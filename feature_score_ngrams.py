
import nltk
from nltk.corpus import stopwords
import string
import operator
from collections import OrderedDict
from textblob import TextBlob
from textblob import Blobber
from textblob.taggers import PatternTagger, NLTKTagger
import os
import re

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
reviewContent = []
alpha = 0.6
def getList():
    # reading from the created file "modified.txt"
    with open("modified.txt") as f:
        review = []
        for line in f:
            if line[:3] == "[t]":
                if review:
                    reviewContent.append(review)
                    review = []
                reviewTitle.append(line.split("[t]")[1].rstrip("\r\n"))
            else:
                if "##" in line:
                    x = line.split("##")
                    for i in range(1, len(x)):
                        review.append(x[i].rstrip("\r\n"))
                else:
                    continue
        reviewContent.append(review)

    tb = Blobber(pos_tagger=NLTKTagger())
    nounScores = dict()
    for a in range(len(reviewContent)):
        for i in range(len(reviewContent[a])):
            text = ' '.join([word for word in reviewContent[a][i].split() if word not in stopwords.words("english")])
            text = ''.join(ch for ch in text if ch not in exclude)
            text = nltk.word_tokenize(text)
            x = nltk.pos_tag(text)
            tagList = []
            for e in x:
                if(e[1] == "NN" or e[1] == "JJ"):
                    tagList.append(e)

            # Add the nouns(which are not in the nounScores dict) to the dict
            for e in tagList:
                if e[1] == "NN":
                    if e[0] not in nounScores:
                        nounScores[e[0]] = 0

            # For every adjective, find nearby noun l=0
            for l in range(len(tagList)):
                if(tagList[l][ 1] == "JJ"):
                    check = 0
                    j = 0
                    k = 0
                    for j in range(l + 1, len(tagList)):
                        if(tagList[j][ 1] == "NN"):
                            check = 1
                            break
                    ct = 0
                    if(l > 0):
                        if j == 0:
                            j = len(tagList)
                        for k in range(l - 1, 0, -1):
                            if ct == 4:
                                break
                            ct += 1
                            if(tagList[k][ 1] == "NN"):
                                if(j != len(tagList)):
                                    nounScores[tagList[min(j, k)][0]] += 1
                                else:
                                    nounScores[tagList[k][0]] += 1
                                break
                    elif check == 1:
                        nounScores[tagList[j][0]] += 1

    nounScores = OrderedDict(sorted(nounScores.items(), key=operator.itemgetter(1)))
    nouns = []
    for key, value in nounScores.items():
        if value >= 3:
            nouns.append(key)
    return nouns

def intersect(a, b):
    return list(set(a) & set(b))


def rankFeatures(adj_scores, features, reviewContent):
    pos_review_index = dict()
    neg_review_index = dict()
    neut_review_index = dict()

    global_noun_scores = dict()
    global_noun_adj_count = dict()

    for a in range(len(reviewContent)):

        reviewContent[a] = reviewContent[a].split()
        review_noun_scores = dict()
        review_noun_adj_count = dict()
        line_words = reviewContent[a]

        line_words = ' '.join([apostropheList[word] if word in apostropheList else word for word in line_words])
        line_words = ''.join(ch for ch in line_words if ch not in exclude)
        line_words = re.sub(r' [a-z][$]? ', ' ', line_words)
        line_words = [word for word in line_words.split() if(word not in stopwords.words("english") and not word.isdigit()) and len(word) > 2]

        for wordIndex in range(len(line_words)):
            word = line_words[wordIndex]

            if word in adj_scores:
                score = adj_scores[word]

                if (wordIndex - 2 >= 0):

                    phrase = line_words[wordIndex - 2] + " " + line_words[wordIndex - 1] + " " + line_words[
                        wordIndex]

                    if ((TextBlob(phrase).sentiment.polarity * score) < 0):
                        score *= -1
                elif (wordIndex - 1 >= 0):

                    phrase = line_words[wordIndex - 1] + " " + line_words[wordIndex]

                    if ((TextBlob(phrase).sentiment.polarity * score) < 0):
                        score *= -1

                closest_noun = find_closest_noun(wordIndex, line_words, features)
                if (closest_noun is None):
                    continue

                if (closest_noun in review_noun_scores):
                    review_noun_scores[closest_noun] += score
                else:
                    review_noun_scores[closest_noun] = score

                if (closest_noun in global_noun_scores):
                    global_noun_scores[closest_noun] += score
                else:
                    global_noun_scores[closest_noun] = score

                if (closest_noun in review_noun_adj_count):
                    review_noun_adj_count[closest_noun] += 1
                else:
                    review_noun_adj_count[closest_noun] = 1

                if (closest_noun in global_noun_adj_count):
                    global_noun_adj_count[closest_noun] += 1
                else:
                    global_noun_adj_count[closest_noun] = 1

        total_score = sum(review_noun_scores.values())
        total_adj = sum(review_noun_adj_count.values())

        if (total_adj == 0):
            review_score = 0
        else:
            review_score = total_score / float(total_adj)

        avg_score = review_score

        # Incase both title_score and review_scores are 0's, then ignore that review
        if (avg_score == 0):
            neut_review_index[a] = avg_score
            continue

        if (avg_score > 0):
            pos_review_index[a] = avg_score
        else:
            neg_review_index[a] = avg_score

    avg_feature_score = dict()
    for noun in global_noun_scores:
        avg_feature_score[noun] = global_noun_scores[noun] / float(global_noun_adj_count[noun])
    avg_feature_score = sorted(avg_feature_score.items(), key=operator.itemgetter(1), reverse=True)

    pos_review_index = OrderedDict(sorted(pos_review_index.items(), key=operator.itemgetter(1), reverse=True))
    neg_review_index = OrderedDict(sorted(neg_review_index.items(), key=operator.itemgetter(1)))

    posPredIndex = []
    negPredIndex = []
    neutPredIndex = []

    # Gather the review index only (not score) from dict
    for i, j in pos_review_index.items():
        posPredIndex.append(i)

    for i, j in neg_review_index.items():
        negPredIndex.append(i)

    for i, j in neut_review_index.items():
        neutPredIndex.append(i)

    # Remove the temp file
    os.remove("modified.txt")
    return posPredIndex, negPredIndex, neutPredIndex, avg_feature_score

def find_closest_noun(wordIndex, line_words, features):
	ptr = 1
	while(ptr <= 3):
		if(wordIndex + ptr < len(line_words) and line_words[wordIndex + ptr] in features):
			return line_words[wordIndex + ptr]
		elif(wordIndex - ptr >= 0 and line_words[wordIndex - ptr] in features):
			return line_words[wordIndex - ptr]
		else:
			ptr += 1


