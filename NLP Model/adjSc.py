
from textblob import TextBlob
import operator
from collections import OrderedDict


def getScore(adjList):
    adjScores = dict()

    for i in adjList:
        blob = TextBlob(i)
        if (blob.sentiment.polarity != 0):
            adjScores[i] = blob.sentiment.polarity

    adjScores.update((x, 4 * y) for x, y in adjScores.items())
    adjScores = OrderedDict(
            [('awesome', 4.0), ('excellent', 4.0), ('wonderful', 4.0), ('ideal', 3.6), ('incredible', 3.6), ('super', 3.8), ('good', 2.3), ('perfect', 4), ('okay', -0.2), ('ok', -0.3),
             ('beautiful', 3.4), ('great', 3.2), ('happy', 3.2), ('bright', 2.8000000000000003), ('nice', 2.4), ('adorable', 2), ('superb', 2), ('worst', -2), ('poor', -1.6),
             ('love', 2.0), ('creative', 2.0), ('able', 2.0), ('satisfied', 2.0), ('pleased', 2.0), ('decent', 0.8), ('elegant', 0.6), ('cute', 0.5),
             ('sophisticated', 2.0),
             ('easy', 1.7333333333333334), ('fantastic', 1.6), ('advanced', 1.6), ('fabulous', 1.6), ('light', 1.6),
             ('comfortable', 1.6), ('available', 1.6), ('clean', 1.4666666666666668), ('quick', 1.3333333333333333),
             ('super', 1.3333333333333333), ('worth', 1.2), ('powerful', 1.2), ('first', 1.0),
             ('positive', 0.9090909090909091),
             ('ready', 0.8), ('real', 0.8), ('reasonable', 0.8), ('fast', 0.8), ('much', 0.8),
             ('main', 0.6666666666666666),
             ('high', 0.64), ('live', 0.5454545454545454), ('clear', 0.4000000000000001), ('flat', -0.1),
             ('minor', -0.2),
             ('single', -0.2857142857142857), ('remote', -0.4), ('partially', -0.4), ('extreme', -0.5), ('sharp', -0.5),
             ('average', -0.6), ('heavy', -0.8), ('flaw', -0.8), ('harsh', -0.8), ('raw', -0.9230769230769231),
             ('small', -1.0),
             ('hard', -1.1666666666666667), ('slow', -1.2000000000000002), ('serious', -1.3333333333333333),
             ('disappointing', -2.4), ('bad', -2.7999999999999994)])

    adjScores = sorted(adjScores.items(), key=operator.itemgetter(1), reverse=True)
    return (OrderedDict(adjScores))


