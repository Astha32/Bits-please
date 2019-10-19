import os
import feature_extraction
import adjSc
from texttable import Texttable
import feature_score_ngrams
import random
import review_summarizer
import pandas as pd

# url = "http://m4y4nk.net/hackabit/redmi.txt"
c = pd.read_csv('redmi.txt', error_bad_lines=False)
c.to_csv(r'redmi.txt', header=None, index=None, sep=' ', mode='a')
filename = 'redmi.txt'

posCount = 0
negCount = 0
neutCount = 0

#Arrays holding the index of the reviews
posActIndex = []
negActIndex = []
neutActIndex = []

reviewContent = []
ratingArray = []
with open(filename) as f:
    review = []
    i = 0
    for line in f:
        l1 = line[1:-1]
        l1 = l1.strip()
        reviewContent.append(l1.rstrip("\r\n"))
        a = random.randint(0, 5)
        ratingArray.append(a)
        if a > 2.5:
            posCount += 1
            posActIndex.append(i)
        elif a < 2.5:
            negCount += 1
            negActIndex.append(i)
        else:
            neutCount += 1
            neutActIndex.append(i)
        i += 1

feature_extraction.fileCreation(reviewContent, filename)
print(reviewContent)
adjDict = feature_extraction.findFeatures(reviewContent, filename)
featureListall = feature_score_ngrams.getList()
featureList = featureListall[0]
adjNounPairs = featureListall[1]
adjScores = adjSc.getScore(adjDict)

posPredIndex, negPredIndex, neutPredIndex, avgFeatScore = feature_score_ngrams.rankFeatures(adjScores, featureList, reviewContent)
outputDir = "./Results_" + filename
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

print(adjDict)
print(adjDict)
print(featureList)

with open(outputDir + "/positiveReviews.txt", "w") as filePos:
    for i in posPredIndex:
        for k in range(len(reviewContent[i])):
            filePos.write(reviewContent[i][k])
            filePos.write(" ")
        filePos.write("\n")

#Write the predicted negative reviews to a file
with open(outputDir + "/negativeReviews.txt", "w") as fileNeg:
    for i in negPredIndex:
        for k in range(len(reviewContent[i])):
            fileNeg.write(reviewContent[i][k])
            fileNeg.write(" ")
        fileNeg.write(" \n")

#Write the predicted neutral reviews to a file
with open(outputDir + "/neutralReviews.txt", "w") as fileNeut:
    for i in neutPredIndex:
        for k in range(len(reviewContent[i])):
            fileNeut.write(reviewContent[i][k])
            fileNeut.write(" ")
        fileNeut.write(" \n")

#Write the predicted neutral reviews to a file
with open(outputDir + "/featureScore.txt", "w") as fileFeat:
    t = Texttable()
    lst = [["Feature", "Score"]]
    for tup in avgFeatScore:
        lst.append([tup[0], tup[1]])
    t.add_rows(lst)
    fileFeat.write(str(t.draw()))

print("The files are successfully created in the dir '" + outputDir + "'")
filespath1 = outputDir + "/negativeReviews.txt"
filespath2 = outputDir + "/positiveReviews.txt"
filespath3 = outputDir + "/neutralReviews.txt"

print("Good things people said about the product::")
print(review_summarizer.summary(filespath2))

print("Bad things people said::")
print(review_summarizer.summary(filespath1))

print("Neutral views of people::")
print(review_summarizer.summary(filespath3))

#Evaluation metric
PP = len(set(posActIndex).intersection(set(posPredIndex)))
PNe = len(set(posActIndex).intersection(set(negPredIndex)))
PN = len(set(posActIndex).intersection(set(neutPredIndex)))

NeP = len(set(negActIndex).intersection(set(posPredIndex)))
NeNe = len(set(negActIndex).intersection(set(negPredIndex)))
NeN = len(set(negActIndex).intersection(set(neutPredIndex)))

NP = len(set(neutActIndex).intersection(set(posPredIndex)))
NNe = len(set(neutActIndex).intersection(set(negPredIndex)))
NN = len(set(neutActIndex).intersection(set(neutPredIndex)))

#Draw the confusion matrix table
t = Texttable()
t.add_rows([['', 'Pred +', 'Pred -', 'Pred N'], ['Act +', PP, PNe, PN], ['Act -', NeP, NeNe, NeN], ['Act N', NP, NNe, NN]])
print("Evaluation metric - Confusion matrix:")
print("=====================================")
print("Dataset source:", filename)
print(t.draw())