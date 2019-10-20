# Review summary based on Features
The project gives the performance score of the various features of the product along with an overall summary of the postive and negative reviews generated with the help of natural language processing and text mining techniques.

### Motivation
1.In recent years, the number of reviews/opinions about a product have seen a tremendous growth which makes it tedious for ​consumers or merchants ​on the web e-commerce to identify the useful, obsolete or substandard features of a particular product. A summary might  help the user to get an general opinion quickly.
2.When a user plans to purchase a product,he maybe interested in a particular feature. For example, a college student may look for a good processor in a phone, a traveller may be interested in camera and a busy tycoon may go for a long lasting battery. Thus, the score and summary of the peformance of the features will be useful for them. This will also help the producers to increase their sales by improving underperforming features of the product.

### Tech Approach
1. NLP model : An opinion mining and ranking algorithm made that classifies the reviews as positive,negative and neutral opinion based on rating and also identifies the features-opinion pairs with the help of tokenization, pos tagging, frequency distribution analysis. The classified reviews tokenized and summary is generated based on their frequency.

2. Web Interface : A flask app created where product name will be entered in a search engine and it will scrap reviews from e-commerce websites for the corresponding product and process it through our model and show the results obtained.


### Getting Started
1. Clone this repository
2.Install nltk==3.2.2, numpy==1.12.1, pyenchant==1.6.8, six==1.11.0, textblob==0.12.0, texttable==1.2.1 before running the code
using the command : pip install nltk==3.2.2
3. Run the command : python3 main.py
4. Open localhost '127.0.0.1:5000' in the browser to view the search engine

### References
- [Research paper by Akkamahadevi R Hanni, Mayur M Patil](https://www.researchgate.net/publication/306285022_Summarization_of_Customer_Reviews_for_a_Product_on_a_website_using_Natural_Language_Processing)
- [NLTK official website](https://nltk.org)
- [POS_Tagger](https://nlpforhackers.io/training-pos-tagger/)
 
