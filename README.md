#-*-coding:utf-8-*-
# Analysis of South Korean Journalism Framing on COVID-19 and 21st South Korea General Election:
## Big Data, Word Embeddings, and Sentiment Analysis Leveraging

Glossary
코퍼스 = 말뭉치 = 분석하려는 대상, 문서, dataset
코퍼스 = 보통 여러 단어들로 이루어진 문장. 한가지 언어로 이루어진 코퍼스 = 단일 언어 코퍼스 (monolingual)
morph = 형태- 형태소(morpheme)의 구체적인 표형

####  Stemming
##### Stemming is the extraction of morphological analysis

#### • TF-IDF (Term Frequency – Inverse Document Frequency) 
#####  The frequency of words in a specific document is proportional to the increase, and the TF-IDF value increases as the number of documents containing the word for all documents decreases. Hence, The bigger the DF value, the lower the weight value of TF-IDF, the reciprocal value of the DF value becomes the IDF.
##### - TF-IDF = TF * IDF = TF (t, d) * IDF (t, D)
##### - TF (Term Frequency) The frequency of specific words extract in a document
##### - DF (Document Frequency) The frequency of document on specific word extract while in all documents
##### - IDF (Inverse Document Frequency) The frequency of document on specific word extract while in all documents
![alt IDF](IDF.png "IDF")
![alt IDF, Prevent denominator equals 0](IDF_.png "IDF, Prevent denominator equals 0")