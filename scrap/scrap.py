#!/usr/bin/env python

from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uRqst
from urllib.parse import urljoin

web_url = "https://www.flipkart.com/redmi-note-7-pro-space-black-64-gb/product-reviews/itmfhvuexbzu6fjd?pid=MOBFHUQ4ZPZX89G6&page=4"



def make_soup(web_url):

	http_client_HTTPResponse = uRqst(web_url)
	web_html = http_client_HTTPResponse.read()
	http_client_HTTPResponse.close()
	beautiful_soup = soup(web_html, "html.parser")
	return beautiful_soup


def fetch_url():

	beautiful_soup = make_soup(web_url)
	for comments in beautiful_soup.findAll("div", {"class": "qwjRop"}):
		comments.split(span)[0]
		print (comments, end ="\n")




	# for tag in div_tags:
	# 	comment = tag.find_all("div", {"class": ""})
	# 	print(comment.text)


































	# links = [each.get('href') for each in a_tags]
	# for each in links:
	# 	if(each[0:4] == "http" or each[0:5] == "https"):
	# 		print(each)
	# 	else:
	# 		full_link = urljoin(web_url,each)
	# 		print(full_link)

	# print(str(len(a_tags))+" Url links found..." )
	# ans = beautiful_soup.find('div' , attrs={'class': 'qwjRop'})
	# print(ans)



if __name__ == '__main__':
	fetch_url()

