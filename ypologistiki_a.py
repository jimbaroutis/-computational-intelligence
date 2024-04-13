import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt




# A1.α
dtype = {
    'names': ('id', 'text', 'metadata', 'region_main_id', 'region_main', 'region_sub_id', 'region_sub', 'date_str', 'date_min', 'date_max', 'date_circa'),
    'formats': ('i4', 'U500', 'U500', 'i4', 'U100', 'i4', 'U100', 'U50', 'f8', 'f8', 'f8')
}

dataset = np.loadtxt('iphi2802.csv', delimiter='\t', dtype=dtype, skiprows=1, encoding='utf-8')


text = dataset['text'].tolist()
date_min = dataset['date_min'].tolist()
date_max = dataset['date_max'].tolist()

# create the vocabulary
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=1000)

# tokenize and build vocab
vectorizer.fit(text)

# summarize

#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)

tokens = vectorizer.get_feature_names_out()
# print(tokens)
# Α1.β -Νormalize data [-1,1]

min_val = np.min(vectorizer.idf_)
max_val = np.max(vectorizer.idf_)
norm_values = 2 *( (vectorizer.idf_ - min_val) / (max_val - min_val) ) -1

print('\n')
#print(norm_values)

#output date_min
min_val = np.min(date_min)
max_val = np.max(date_min)
norm_date_min = 2 *( (date_min- min_val) / (max_val - min_val) ) -1

#print(norm_date_min)
#output date_max
min_val = np.min(date_max)
max_val = np.max(date_max)
norm_date_max = 2 *( (date_max- min_val) / (max_val - min_val) ) -1

#print(norm_date_max)
# Α1.γ -Split the data to training and testing data 5-Fold

term_index_map = {term: i for i, term in enumerate(tokens)}

text_matrix = np.zeros((len(text), len(tokens)))

#idf_value = vectorizer.idf_[vectorizer.vocabulary_["εποιησε"]]
#print(idf_value)

#idf_value = vectorizer.idf_[373]
#print(idf_value)

# mapping processing
for i, doc in enumerate(text):

    tokens = doc.split()

    for token in tokens:
        
        if token in term_index_map:

            term_index = term_index_map[token]
              
            text_matrix[i, term_index] = norm_values[term_index]

#print("Transformed Corpus Matrix shape:", text_matrix.shape)
#np.set_printoptions(threshold=sys.maxsize)

# combine the two output arrays
date_min_max = np.vstack((norm_date_min, norm_date_max)).T
#print(date_min_max)
# print(date_min_max.shape)
# print(text_matrix.shape)

# Split the data to training and testing data 5-Fold

kfold = KFold(n_splits=5, shuffle=True)

rmseList = []
rrseList = []

#early_stopping = EarlyStopping(monitor='rmse', patience=500, restore_best_weights=True, mode ='min')


# loop
for i, (train, test) in enumerate(kfold.split(text_matrix)):
  
# A2

    # Create model
    model = Sequential()

    model.add(Dense(128, activation="tanh", input_dim=1000))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation ="tanh", input_dim =128))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="linear", input_dim =128))
   
    # compile model
    def rmse(y_true, y_pred):
     errors = tf.where(
       
        y_pred[:, 0] < y_true[:, 0],
        y_true[:, 0] - y_pred[:,0],
        tf.where(
            y_pred[:, 1] > y_true[:, 1],
            y_pred[:, 1] - y_true[:, 1],
            tf.zeros_like(y_pred[:, 0])
        )
    )
     return tf.sqrt(tf.reduce_mean(tf.square(errors)))
    
    keras.optimizers.SGD(learning_rate=0.001, momentum=0.6, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])
    
    # Fit model
    m_f = model.fit(text_matrix[train], date_min_max[train], epochs=500, batch_size=500,verbose=0)
 
    # Plot model
    rmse_history= m_f.history['rmse']
    epochs = range(1, len(rmse_history) + 1)
    plt.plot(epochs, rmse_history, 'b', label='Training RMSE')
    plt.title('Training RMSE per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # Evaluate model
    scores = model.evaluate(text_matrix[test], date_min_max[test], verbose=0)
    rmseList.append(scores[1])
    print("Fold :", i, " RMSE:", scores[1])  

print("RMSE: ", np.mean(rmseList))

