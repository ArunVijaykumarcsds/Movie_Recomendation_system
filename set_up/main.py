import numpy as np 
import pandas as pd
import warnings 
# Reading csv files into dataframe
warnings.filterwarnings("ignore")
credits = pd.read_csv(
    "credits.csv",
    encoding="latin1",
    encoding_errors="ignore"   # works in newer pandas
)

movies = pd.read_csv('movies.csv')
movies.info()
credits.info()
credits.head()
movies = movies.merge(credits,on='title')
movies.head()
movies['original_language'].value_counts()
movies.info()
movies = movies[['genres','id','keywords','title','overview','cast','crew']]
movies.head(1)
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.head()
movies.iloc[0]['keywords']
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies.iloc[19].genres
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'][0]
def convert3(obj):
    L = []
    ct = 0
    for i in ast.literal_eval(obj):
        if ct!=3:
            L.append(i['name'])
            ct = ct+1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert3)
movies['cast'][0]
movies['crew'][0]
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# Removing spaces between words
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'][0]
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# Now we will use stemming using nltk library to remove same words like love, loving, loved
# new_df.iloc[3091]
# 29    when bond' latest assign goe grave wrong and a...
# Name: tags, dtype: object

# !pip3 install nltk
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('walking')
def stem(text):
    y = []
    
    # Converting string into list
    
    for i in text.split():
        y.append(ps.stem(i))
        
    # Again converting list into string
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)
# Now we will do vectorization of data in tags column            
# First we'll combine all the tags and calculate 5000 words whose frequency is maximum
# !pip3 install scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors.shape #vector containing top  5000 features from every 4806 rows
vectors[0] # Vector of movie Avatar
cv.get_feature_names_out()
stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')
# Now we have 4806 vectors (one for each movie)       
# Now we will calculate distance between two vectors to find similarity between movies          
# We will calculate cosine distance (angle b/w vectors) rather than euclidean distance           
# As the dimension of data increases, the performance of euclidean distance decreases        
from sklearn.metrics.pairwise import cosine_similarity
#This calculates the cosine similarity between vectors which tells how similar each vector to 
#each other this will help to get similar movies w.r.t to our feed
cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)

print(similarity.shape)
lst=[1,2,13,4,5,6]
x=sorted(lst,reverse=True)
print(x)
new_df.iloc[2]['title']
movie_index = new_df[new_df['title'] == 'Spectre'].index[0]
print(movie_index)
distances = similarity[movie_index]
print(distances)
print(list(enumerate(distances)))

# movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]

# print(movies_list)

#for i in movies_list:
#       print(new_df.iloc[i[0]].title)
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
      
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
# above code will give error if similar movies are not presnt       
# so the below code will handel that error using try and except block
def recommend(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        for i in movies_list:
            print(new_df.iloc[i[0]].title)
                    
    except IndexError:
        print(movie , 'not found')
new_df['title'] == 'Avatar'
new_df[new_df['title'] == 'Avatar']
recommend('Avatar')
recommend('Batman')
recommend('Iron Man')
recommend('Spider-Man 3')
