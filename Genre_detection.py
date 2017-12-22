

#only for jupyter notebooks
get_ipython().magic('matplotlib inline')
#rest are same
import seaborn
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import sklearn, pandas
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)




#features for storing the mfcc values and labels store the category to which the genre belongs like 0 for blues , 1 for disco and so on
features = []
labels = []



for i in range(11 , 100):  #reading music from dataset
    var=i
    filename = "genres/blues/"+"blues.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(0)



for i in range(11 , 100):
    var=i
    filename = "genres/disco/"+"disco.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler= sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(1)




for i in range(12 , 15):
    var=i
    filename = "genres/country/"+"country.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(2)


# In[6]:

for i in range(11 , 100):  
    var=i
    filename = "genres/classical/"+"classical.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(3)




for i in range(11 , 100):  
    var=i
    filename = "genres/hiphop/"+"hiphop.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(4)


# In[8]:

for i in range(11 , 100):  
    var=i
    filename = "genres/jazz/"+"jazz.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(5)



for i in range(11 , 100):  
    var=i
    filename = "genres/metal/"+"metal.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(6)



for i in range(11 , 100):  
    var=i
    filename = "genres/pop/"+"pop.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(7)


# In[11]:

for i in range(11 , 100):  
    var=i
    filename = "genres/reggae/"+"reggae.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(8)



for i in range(11 , 100):  
    var=i
    filename = "genres/rock/"+"rock.000"+str(var)+".au"
    x_file , sr_file = librosa.load(filename , duration = 30)
    mfcc_x= librosa.feature.mfcc(x_file, sr = sr_file, n_mfcc = 12).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_x_scaled=scaler.fit_transform(mfcc_x)
    #print(mfcc_x_scaled.shape)
    features.append(mfcc_x_scaled)
    for j in range(len(mfcc_x_scaled)):
        labels.append(9)

features = numpy.array(features) #to create numpy array
labels = numpy.array(labels)



features = features.reshape(features.shape[0]*features.shape[1],features.shape[2]) #reshaping 3-d numpy array to 2-d


#using svm classifier
model = sklearn.svm.SVC()
model.fit(features, labels)
score_svm = model.score(features, labels)




#using gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=1000, learning_rate=1.5, max_depth=1, random_state=0)
model.fit(features, labels)
score_gradBoost = model.score(features, labels)



#testing on a random song
x , sr_x = librosa.load('genres/country/country.00044.au',duration=10, offset=80)
mfcc_x= librosa.feature.mfcc(x, sr = sr_x, n_mfcc = 12).T
scalers = sklearn.preprocessing.StandardScaler()
x_scaled=scalers.fit_transform(mfcc_x)


#finding out the label having maximum occurences in predicted_labels
predicted_labels = model.predict(x_scaled)
numpy.argmax([(predicted_labels == c).sum() for c in (0, 1, 2)])


#end
