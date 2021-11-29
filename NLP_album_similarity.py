from selenium import webdriver
import warnings
warnings.filterwarnings(action = 'ignore')
import time
import csv
import pandas as pd
import re
import spacy
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from gensim import models
from scipy import spatial
import gensim.downloader as api
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


# Below code till the next comment is written to scrape the data from the website using selenium and store it into a csv file in the desired format

#_____________________________________________________________________________________________________________________
# Beautiful soup hasn't been used to scrape the data due to the presence of java script on the page

search_query = 'https://www.pastemagazine.com/music/best-selling-albums/the-best-selling-albums-of-all-time/'
driver = webdriver.Chrome(executable_path='C:/chromedriver')  #Need to install the chrome driver for this code
driver.get('https://www.pastemagazine.com/music/best-selling-albums/the-best-selling-albums-of-all-time/')
time.sleep(2)
final_lst1=[]
final_lst2=[]
try:
    main = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'article-detail-container'))
    )
    #print (main.text)
    albums=main.find_elements_by_tag_name('h2')       # this will give you all the data present in the h2 tag
    description=main.find_elements_by_tag_name('p')   #this iwll give you all the data present in the paragraph(p) tag
    headerList = ['id', 'artist', 'album']
    with open('NLP_data.csv', 'w', newline='',encoding='utf-8-sig') as f:
        writer = csv.writer(f)
    
        # write the header
        writer.writerow(headerList)
    
        for album in albums:
           lst1=album.text.split(':')
           if '' not in lst1 and (len(lst1)>2 or len(lst1)==2):
               
               #print (lst1[0].split('.')[1])  # this will be artist
               artist=lst1[0].split('.')[1]
               #print (lst1[1])  # this will be album
               album=lst1[1]
               lst1[0]=re.sub(' +','-',lst1[0])
               lst1[1]=re.sub(' +','-',lst1[1])
               string=lst1[0]+lst1[1]
               string=string.replace('.','')
               string=string.lower()
               id=string
               for var in lst1:
                   if '\n' in var:
                      lst1.remove(var)
                      year=lst1[len(lst1)-1]
                   else:
                       year="0"
               writer.writerow([id,artist,album])
               final_lst1.append([id,artist,album,year])
               #print (string)   #  this is the id column of csv
    headerList2=['decade', 'sales_total', 'sales_us', 'description', 'author']            
    with open('NLP_data2.csv', 'w', newline='',encoding='utf-8-sig') as f:
        writer = csv.writer(f)
    
        # write the header
        writer.writerow(headerList2)
    
        for desc in description:
                #print (desc.text)
                if "Label:" in desc.text and "U.S. sales:" in desc.text:
    
                    lst2=desc.text.split('\n')

                    for var in lst2:
                       
                        if 'Year:' in var:
                            year=var.split(':')[1]
                            year=int(year)%100   # Converting year into decade
                            if year < 10:
                                year="00s"
                            elif year < 20:
                                year="10s"
                            elif year < 30:
                                year="20s"  
                            elif year < 40:
                                year="30s"  
                            elif year < 50:
                                year="40s"
                            elif year < 60:
                                year="50s"  
                            elif year < 70:
                                year="60s" 
                            elif year < 80:
                                year="70s"
                            elif year < 90:
                                year="80s"  
                            elif year < 100:
                                year="90s"
                           
                        elif 'Total certified sales:' in var:
                            sales_total=var.split(':')[1].split('(')[0]
                            sales_total= (int(sales_total.split(' ')[1])*1000000)  #converting number from words to figures(since million is present throughout the data, I have multiplied by 10^6)
                        elif 'Total sales:' in var:
                            
                            sales_total=var.split(':')[1].split('(')[0]
                            sales_total= (int(sales_total.split(' ')[1])*1000000)
                        
                        elif 'U.S. sales:' in var:
                           
                            sales_us=var.split(':')[1]
                            sales_us= (int(sales_us.split(' ')[1])*1000000)
                        else:
                          if 'Label:' not in var:
                            description=var   #can't remove author from here coz of big dash
                            author=var.split(" ")
                            auth=author[len(author)-2]+" "+ author[len(author)-1]
                            auth=auth.split('.')
                            if len(auth)>1:
                                auth=auth[1]   
                            else:
                                auth=auth[0]   
                           
                            
                    writer.writerow([year,sales_total,sales_us,description,auth])
        
                    #writer.writerow([year,sales_total,sales_us,description])
                
     
    with open('NLP_data.csv',encoding='utf-8-sig') as in_1, open('NLP_data2.csv',encoding='utf-8-sig') as in_2, open('final.csv', 'w',newline='',encoding='utf-8-sig') as out:
        reader1 = csv.reader(in_1)
        reader2 = csv.reader(in_2)
        writer = csv.writer(out)
        for row1, row2 in zip(reader1, reader2):
            if row1[0] and row1[1] and row1[2] and row2[0] and row2[1] and row2[2] and row2[3] and row2[4]:
                writer.writerow([row1[0], row1[1],row1[2], row2[0], row2[1], row2[2], row2[3], row2[4]])
            
                

finally:
    driver.quit()


# Reading the final created data and performing pre-processing

df = pd.read_csv('final.csv',encoding='utf-8-sig')
nlp = English()
#sbd = nlp.create_pipe('sentencizer')
#nlp.add_pipe(sbd)
main_lst=[]
main_token_lst=[]
for text in df['description']:
    #text.replace(',','')
    text_lst=text.split('.')
    text_lst=text_lst[:len(text_lst)-1]  #removing author from description
    #print (text_lst)
    for sent in text_lst:
        sent1=sent.lower()  # converting everything to lower case
        my_doc = nlp(sent1)
        token_list = []
        for token in my_doc:
            if token.is_stop==False:     # removing stop words
                token_list.append(token.lemma_) #stemming
        
        if len(token_list) > 19:  #removing sentences which contain 20 tokens or more
            text_lst.remove(sent)

        
    #main_lst.append(text_lst)
    
    s=' '.join(map(str, text_lst))
    main_lst.append(s)
    
               
    #print(text_lst)  #now contains only sentences with less than 20 tokens
    #print ("\n")
df['new_desc']=pd.Series(main_lst)


        
    
df.to_csv('final.csv', encoding='utf-8-sig')  # final file with new description reduced by tokenization

print ("Pearson Correlation between total and US sales is" +" "+ str(df.sales_total.corr(df.sales_us, method="pearson")))  #Pearson correlation btw total and US sales
print ("Spearman Correlation between total and US sales is"+ " " + str(df.sales_total.corr(df.sales_us, method="spearman"))) #spearman correlation

print ("_________________________") 

# Using Spacy to perform entity recognition
NER = spacy.load("en_core_web_sm")  
count_dic_lst=[]
count_dict={}
dict={}

for text in df['new_desc']:
    entity_text=NER(str(text)) 
    for word in entity_text.ents:
        if word.label_ in dict:
            if word.text not in dict[word.label_]:
                dict[word.label_].append(word.text)   #word.label will be entity class and #word.text is entity mention
                count_dict[word.label_]+=1
        else:
            dict[word.label_]=[]
            count_dict[word.label_]=[]
            dict[word.label_].append(word.text)
            count_dict[word.label_]=1
count_dict=sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
#count_dic_lst.append(count_dict)
print ("Here are the top-5 most frequent entity classes in the entire dataset") # top 5 most freq entity classes
for i in range(5):
    print (count_dict[i])
    
print ("_________________________")
#_________________________________



dic_lst=[]
for text in df['new_desc']:
    entity_text=NER(str(text)) 
    dict={}
    for word in entity_text.ents:
        if word.label_ in dict:
            if word.text not in dict[word.label_]:
                dict[word.label_].append(word.text)   #word.label will be entity class and #word.text is entity mention
                
        else:
            dict[word.label_]=[]
            dict[word.label_].append(word.text)

    dic_lst.append(dict)
print (dic_lst)  # storing entity mentions in the dictionary for each album description

print ("_________________________")

#__________________________________

dict={}
for text in df['new_desc']:
    entity_text=NER(str(text)) 

    for word in entity_text.ents:
        if word.text in dict:
            
                dict[word.text]+=1   
                
        else:
            dict[word.text]=1


dict=sorted(dict.items(), key=lambda x: x[1], reverse=True)
print ("Here are the top-10 most frequent entity mentions in the entire dataset") # top 10 most freq entity mentions
for i in range(10):
    print (dict[i])

#df['entity_dictionary']=pd.Series(dic_lst)
#df.to_csv('final.csv', encoding='utf-8-sig', index=False)  # dict of each entry (entity mentions)
    
#Build a system for comparing album descriptions using different algorithms 

#Using Jaccard similarity of entity classes of 2 albums  
"""    
txt=input("Enter the phrase to find the most related albums")   
entity_txt=NER(str(txt)) 
input_entity_classes=[]
for word in entity_txt.ents:
    if word.label_ not in input_entity_classes:
        input_entity_classes.append(word.label_)
print (input_entity_classes)
entity_lst=[]
for text in df['new_desc']:
    lst=[]
    entity_text=NER(str(text)) 
    for word in entity_text.ents:
        if word.label_ not in lst:
            lst.append(word.label_)
    entity_lst.append(lst)

for val in entity_lst:
    s1 = set(val)
    s2 = set(input_entity_classes)
    print (float(len(s1.intersection(s2)) / len(s1.union(s2))))


"""
#Using Cosine similarity
main_token_lst=[]
sbd = nlp.create_pipe('sentencizer')
nlp.add_pipe(sbd)
for text in df['description']:  #pre-processing the data
        text=text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text ) 
        text = re.sub(r'\s+', ' ', text)  # removing special characters
        my_doc = nlp(text)
        token_list = []
        for token in my_doc:
             if token.is_stop==False:     # removing stop words
                token_list.append(token.text)
        token_list=token_list[:len(token_list)-2]
        main_token_lst.append(token_list)
        for i in range(len(main_token_lst)):
            main_token_lst[i] = [w for w in main_token_lst[i] if w not in stopwords.words('english')]  # remove stop words


#model = Word2Vec(main_token_lst,  vector_size=25, epochs=100,min_count=1) #train the model using word2vec
#vocabulary = list(model.wv.index_to_key)
#word_vectors = model.wv
#model = api.load("glove-wiki-gigaword-50")  #using pretrained embeddings but not working if key not present
df['cleaned_data']=pd.Series(main_token_lst)
dictionary = gensim.corpora.Dictionary(main_token_lst) #convert each word to a unique id
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in main_token_lst] # number of times each word has occured (basically id of word and its count (bag of words))
tf_idf = gensim.models.TfidfModel(corpus) #words occuring more freq will get less weights
"""
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
"""
sims = gensim.similarities.Similarity('C:/Users/asus/Desktop',tf_idf[corpus],  #building the index
                                        num_features=len(dictionary))


# Give the phrase to find similarity in the demofile2.txt
file2_docs = []
with open ('demofile2.txt', encoding='utf-8') as f:
    text=f.read()
    text=text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text ) 
    text = re.sub(r'\s+', ' ', text)  # removing special characters
    my_doc = nlp(text)
    token_list = []
    for token in my_doc:
        if token.is_stop==False:     # removing stop words
                token_list.append(token.text)
query_doc=token_list
query_doc_bow = dictionary.doc2bow(query_doc)
print (query_doc)
print (query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print('Comparing Result:', sims[query_doc_tf_idf]) 
print ("Here are the top 5 matching albums to your text:\n")
i=25
dict_album_similar={}
for val in sims[query_doc_tf_idf]:
    dict_album_similar[i]=val
    i-=1
dict_album_similar=sorted(dict_album_similar.items(), key=lambda x: x[1], reverse=True)
for i in range(5):
    print(df['album'][25-dict_album_similar[i][0]] + "------"+ "Cosine similarity is...." + str(dict_album_similar[i][1]))

print ("\n")
# DS track__________
print ("DS track.......\n")
for i in range(3):
    print(str(len(df[df['decade']==df['decade'][25-dict_album_similar[i][0]]])/len(df['decade'])*100) + "%" + " "+ "of the total albums have same decade as that of album"+ " "+df['album'][25-dict_album_similar[i][0]])
    print(str(len(df[df['author']==df['author'][25-dict_album_similar[i][0]]])/len(df['author'])*100) + "%" + " "+ "of the total albums have same author as that of album"+ " "+df['album'][25-dict_album_similar[i][0]])
    print("________")








