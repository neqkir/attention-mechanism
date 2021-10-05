##### https://machinelearningmastery.com/the-attention-mechanism-from-scratch/

'''
Within the context of machine translation, each word in an input sentence would be attributed its own query,
key and value vectors. These vectors are generated by multiplying the encoder’s
representation of the specific word under consideration, with three different weight matrices
that would have been generated during training. 

In essence, when the generalized attention mechanism is presented with a sequence of words,
it takes the query vector attributed to some specific word in the sequence and scores it
against each key in the database. In doing so, it captures how the word under consideration
relates to the others in the sequence. Then it scales the values according
to the attention weights (computed from the scores), in order to retain focus on those words
that are relevant to the query. In doing so, it produces
an attention output for the word under consideration.
'''

import numpy as np

# encoder representations of four different words
word_1 = np.array([1, 0, 0])
word_2 = np.array([0, 1, 0])
word_3 = np.array([1, 1, 0])
word_4 = np.array([0, 0, 1])

# generating the weight matrices
np.random.seed(42) # to allow us to reproduce the same attention values
W_Q = np.random.randint(3, size=(3, 3))
W_K = np.random.randint(3, size=(3, 3))
W_V = np.random.randint(3, size=(3, 3))

# generating the queries, keys and values
query_1 = word_1 @ W_Q
key_1 = word_1 @ W_K
value_1 = word_1 @ W_V
 
query_2 = word_2 @ W_Q
key_2 = word_2 @ W_K
value_2 = word_2 @ W_V
 
query_3 = word_3 @ W_Q
key_3 = word_3 @ W_K
value_3 = word_3 @ W_V
 
query_4 = word_4 @ W_Q
key_4 = word_4 @ W_K
value_4 = word_4 @ W_V

####
# considering the first word only

# scoring the first query vector against all key vectors
scores = np.array([np.dot(query_1, key_1), np.dot(query_1, key_2), np.dot(query_1, key_3), np.dot(query_1, key_4)])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  

# computing the weights by a softmax operation
weights = softmax(scores / key_1.shape[0] ** 0.5)

# computing the attention by a weighted sum of the value vectors
attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)
 
print(attention)

####
# considering all words in a matrix fashion


# stacking the word embeddings into a single array
words = np.array([word_1, word_2, word_3, word_4])
 
# generating the weight matrices
np.random.seed(42)
W_Q = np.random.randint(3, size=(3, 3))
W_K = np.random.randint(3, size=(3, 3))
W_V = np.random.randint(3, size=(3, 3))
 
# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V
 
# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

from scipy.special import softmax

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
 
# computing the attention by a weighted sum of the value vectors
attention = weights @ V
 
print(attention)
