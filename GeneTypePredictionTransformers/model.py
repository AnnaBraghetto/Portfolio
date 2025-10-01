import tensorflow as tf
from tensorflow import keras
from keras import layers,models

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.metrics import confusion_matrix



class SinusoidalPositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, max_len, embed_dim):
        '''Initalization of the SinusoidalPositionalEncoding class
        Input:
        max_len = int, maximum length of input sequences
        embed_dim = int, embedding dimension
        Output:
        '''

        #inherit
        super().__init__()
        #save max len
        self.max_len = max_len
        #save embed dim
        self.embed_dim = embed_dim
        #create positional encoding matrix
        self.pos_encoding = self._get_positional_encoding(max_len, embed_dim)


    def _get_positional_encoding(self, max_len, embed_dim):
        '''Create sinusoidal positional encoding matrix
        Input:
        max_len = int, maximum length of input sequences
        embed_dim = int, embedding dimension
        Output:
        pos_encoding = tf.Tensor, shape (1, max_len, embed_dim)
        '''

        #define positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        #define div term
        div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        #prepare positional encoding matrix
        pe = np.zeros((max_len, embed_dim))
        #sinusoidal functions
        pe[:, 0::2] = np.sin(position * div_term)
        #cosinoidal functions
        pe[:, 1::2] = np.cos(position * div_term)
        #add batch dimension
        pe = pe[np.newaxis, ...]  # shape: (1, max_len, embed_dim)

        #return as tf tensor
        return tf.cast(pe, dtype=tf.float32)


    def call(self, x):
        '''Add positional encoding to input tensor
        Input:
        x = tf.Tensor, shape (batch_size, seq_len, embed_dim)
        Output:
        tf.Tensor, shape (batch_size, seq_len, embed_dim)
        '''

        #get seq length
        seq_len = tf.shape(x)[1]

        #return input + positional encoding
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerClassifier():

    def __init__(self,vocab_size, max_len, num_classes):
        '''
        Initialization of the Transformer class.
        Input:
        vocab_size = int, size of the vocabulary
        max_len = int, maximum length of input sequences
        num_classes = int, number of output classes
        Output:
        '''

        #save parameters
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes

    
    def build(self,embed_dim=128,num_heads=4,ff_dim=256,dropout=0.1):
        '''
        Build the Transformer model.
        Input:
        embed_dim = int, embedding dimension
        num_heads = int, number of attention heads
        ff_dim = int, feedforward network dimension
        dropout = float, dropout rate
        Output:
        model = tf.keras.Model, compiled Transformer model
        '''

        #take inputs    
        inputs = layers.Input(shape=(self.max_len,), dtype='int32')

        #embedding layer with mask on padding (id=0)
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=embed_dim, mask_zero=True)(inputs)

        #positional encoding (sinusoidal)
        x = SinusoidalPositionalEncoding(max_len=self.max_len, embed_dim=embed_dim)(x)

        #multi-head self-attention (mask handled automatically)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        #feedforward network
        ffn = layers.Dense(ff_dim, activation='relu')(x)
        ffn_output = layers.Dense(embed_dim)(ffn)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        #global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        #final classification layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        #build model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    

def plot_confusion_matrix(model,X_test,y_test):

    y_probs = model.predict(X_test)             #predict probabilities
    y_pred = np.argmax(y_probs, axis=1)        #convert to class indices

    #compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    #display confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Test Data')
    plt.xlabel('Predicted GeneType')
    plt.ylabel('Actual GeneType')
    plt.tight_layout()
    plt.show()
    plt.show()