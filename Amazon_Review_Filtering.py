

import json
result=[]

#loading the json file

with open('reviews.json', 'r') as f:
    for item in f:
        data = json.loads(item)
        dic={}
        dic['title']=data['reviewerID']
        dic['description']=data['reviewText']
        result.append(dic)


#creating the table of reviews
import pandas as pd
table  = pd.DataFrame(result)



#stop = set(stopwords.words('english'))

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


stop = set(stopwords.words('english'))

#PorterStemmer object
ps = PorterStemmer()


# stemming and stop wprd removal
for i in range(0,1689188):
    new_tokens= [j for j in word_tokenize(table['description'].iloc[i].lower()) if j not in stop]
    new_tokens = [ps.stem(k) for k in new_tokens]
    table['description'].iloc[i]=" ".join(new_tokens)



#removing all reviews with less than 10 words

for i in range(len(table)):
    review = table['description'].iloc[i]
    if(len(review)<10):
        table.drop(table.index[i],inplace = True)
    

#Building the vocabulary of all distinct word
def dist_words():
    data_words=set()
    for i in range(len(table)):
        review = table['description'].iloc[i]
        set.update([word for word in review])
        
def freq(word , review):
    return review.split().count(word)
        
vocabulary = dist_word()
tf_matrix = []

#building the term-frequency vector
for i in range(len(table)):
    review = table['description'].iloc[i]
    tf_vector = [freq(word, review) for word in vocabulary]
    tf_matrix.append(tf_vector)
    
#normalising    
def normalizer(tf_vector):
    a=max(tf_vector)
    return [vec/a for vec in tf_vector]


tf_vector_norm = []

for vec in tf_vector
    tf_vector_norm.append(normalizer(vec))



#building the inverse-document frequency vector

def idf(word):
    for i in range(len(table)):
        num_of_doc =0
        if freq(word , table['description'].iloc[i] ) > 1:
            num_of_doc+=1
    return np.log(len(table) / 1+num_of_doc)

idf_vector = [idf(word) for word in vocabulary]


#computing the tf-idf product for each tf_vector

def build_idf_matrix(vec):
    mat=np.zeros(len(vec),len(vec))
    mat = np.fill_diagonal(mat,vec)
    return mat

idf_mat = build_idf_matrix(idf_vector)

unfinal_matrix = []

for vec in tf_vector:
    prod_vec = np.dot(vec,idf_mat)
    unfinal_matrix.append(prod_vec)
    
final_matrix = []
    
for vec in unfinal_matrix:
    final_matrix.append(normalizer(vec))
    
#computing cosine similarity for all the reviews with respect to all other reviewa

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1,vector2)
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if  magnitude == 0 :
        return 0
    return dot_product/magnitude

comparison_matrix = []

for first_vec in final_matrix:
    similar_array = []
    for second_vec in final_matrix:
        sim_const = cosine_similarity(first_vec , second_vec)
        similar_array.append(sim_const)
    comparison_matrix .append(similar_array)
    
comparison_matrix  = np.array(comparison_matrix)
    
print  (comparison_matrix)
    

