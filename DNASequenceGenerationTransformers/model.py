import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np


class TransformerGenerator():

    def __init__(self,vocab_size, max_len):
        '''
        Initialization of the Transformer class
        Input:
        vocab_size = int, size of the vocabulary
        max_len = int, maximum length of input sequences
        Output:
        '''

        #save parameters
        self.vocab_size = vocab_size
        self.max_len = max_len


    def build(self,d_model=128,num_layers=4,n_heads=8,ff_dim=512,dropout=0.1):
        '''
        Build the Transformer model
        Input:
        d_model = int, dimension of the model
        num_layers = int, number of transformer layers
        n_heads = int, number of attention heads
        ff_dim = int, dimension of the feed-forward network
        dropout = float, dropout rate
        Output:
        model = tf.keras.Model, the compiled Transformer model
        '''

        def create_look_ahead_mask(seq_len):
            '''
            Create a look-ahead mask to mask future tokens in a sequence
            Input:
            seq_len = int, length of the sequence
            Output:
            '''
        
            #upper triangular matrix of 1s above the diagonal
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            return mask  # shape: (seq_len, seq_len)


        #input
        inputs = layers.Input(shape=(self.max_len,), dtype="int32")

        #token embedding + positional encoding
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=d_model, mask_zero=True)(inputs)
        pos_indices = tf.range(start=0, limit=self.max_len, delta=1)
        pos_encoding = layers.Embedding(input_dim=self.max_len, output_dim=d_model)(pos_indices)
        x = x + pos_encoding

        #padding mask
        padding_mask = layers.Lambda(lambda x: tf.cast(tf.not_equal(x, 0), tf.float32))(inputs)
        padding_mask = layers.Lambda(lambda x: x[:, tf.newaxis, tf.newaxis, :])(padding_mask)

        #look-ahead mask
        look_ahead_mask = create_look_ahead_mask(self.max_len)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :] #broadcast

        #combined mask
        combined_mask = layers.Lambda(lambda x: tf.maximum(x[0], 1 - x[1]))([look_ahead_mask, padding_mask])


        #stack of decoder layers
        for _ in range(num_layers):
            #multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model,
                dropout=dropout
            )(x, x, attention_mask=combined_mask)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output) #residual + norm

            #feed-forward network
            ffn = layers.Dense(ff_dim, activation="relu")(x)
            ffn = layers.Dense(d_model)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn) #residual + norm

        #output layer
        outputs = layers.Dense(self.vocab_size, activation="softmax")(x)

        #create model
        self.model = models.Model(inputs=inputs, outputs=outputs)


    def compile(self):
        '''
        Compile the Transformer model with custom loss and optimizer
        Input:
        Output:
        '''

        #custom loss to ignore padding tokens

        def masked_loss(y_true, y_pred):
            '''
            Compute loss ignoring padding tokens (id=0)
            Input:
            y_true = true labels, shape (batch, seq_len)
            y_pred = predicted logits, shape (batch, seq_len, vocab_size)
            Output:
            loss = scalar, mean loss over non-padding tokens
            '''

            #sparse categorical crossentropy
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
            #create mask to ignore padding tokens (id=0)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

            #compute element-wise loss
            loss = loss_fn(y_true, y_pred)  #shape: (batch, seq_len)

            #apply mask to loss
            loss *= mask

            #return mean loss over non-padding tokens
            return tf.reduce_sum(loss) / tf.reduce_sum(mask)

        
        def masked_accuracy(y_true, y_pred):
            '''
            Compute masked accuracy ignoring padding tokens
            Input:
            y_true = true labels, shape (batch, seq_len)
            y_pred = predicted logits, shape (batch, seq_len, vocab_size)
            Output:
            accuracy = scalar, accuracy over non-padding tokens
            '''
            
            #create mask for non-padding tokens
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

            #get predicted token ids
            pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

            #compare with ground truth
            matches = tf.cast(tf.equal(y_true, pred_ids), tf.float32)

            #apply mask
            matches *= mask

            #compute accuracy over non-padding tokens
            return tf.reduce_sum(matches) / tf.reduce_sum(mask)


        #compile model
        self.model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_accuracy])
        #print model summary
        self.model.summary()


    
    def fit(self,X_train,y_train,X_val,y_val):
        '''
        Train the Transformer model
        Input:
        X_train = np.array, training input sequences
        y_train = np.array, training target sequences
        X_val = np.array, validation input sequences
        y_val = np.array, validation target sequences
        Output:
        '''


        #early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',        #metric to monitor
            patience=5,                #stop if no improvement after 5 epochs
            restore_best_weights=True  #restore model weights from best epoch
        )

        #learning rate decay
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',        #metric to monitor
            factor=0.5,                #reduce lr by factor
            patience=3,                #wait 3 epochs before reducing
            min_lr=1e-6                #minimum learning rate
        )

        #fit model
        self.model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),  #provide validation set explicitly
            callbacks=[early_stop, reduce_lr]
        )


    def compute_perplexity(self, X_test, y_test, batch_size=32):
        '''
        Compute perplexity in batches without running out of memory
        Input:
        X_test = np array, test set
        y_test = np array, labels
        batch_size = int, batch size
        Output:
        perplexity = float, perplexity of the model on the test set
        '''

        #prepare to accumulate log probabilities and token counts
        total_log_prob = 0.0
        total_token_count = 0

        #prepare test set
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

        for batch_x, batch_y in test_dataset:
            #predictions: (batch, seq_len, vocab_size)
            preds = self.model(batch_x, training=False).numpy()

            #get token probabilities
            token_probs = preds[np.arange(preds.shape[0])[:, None], np.arange(preds.shape[1]), batch_y.numpy()]

            #mask padding tokens
            non_padding_mask = (batch_y.numpy() != 0)
            valid_token_probs = token_probs[non_padding_mask]

            #avoid log(0)
            valid_token_probs = np.clip(valid_token_probs, 1e-8, 1.0)

            #accumulate
            total_log_prob += np.sum(np.log(valid_token_probs))
            total_token_count += len(valid_token_probs)

        #compute perplexity
        perplexity = np.exp(-total_log_prob / total_token_count)
        return perplexity

            
    def generate_samples(self, seed_seq, temperature=1.0, gen_len=50, num_samples=5):
        '''
        Generate new sequences using the trained model
        Input:
        seed_seq = list of int, seed sequence to start generation
        max_len = int, maximum length of generated sequences
        vocab_size = int, size of the vocabulary
        temperature = float, controls randomness in sampling
        num_samples = int, number of sequences to generate
        Output:
        list of generated sequences (each a list of int)
        '''


        generated_sequences = []

        for _ in range(num_samples):
            generated = seed_seq.copy()

            for _ in range(gen_len - len(seed_seq)):
                input_seq = np.array(generated)[np.newaxis, :]

                #truncate if too long
                if input_seq.shape[1] > self.max_len:
                    input_seq = input_seq[:, -self.max_len:]

                #pad if too short
                if input_seq.shape[1] < self.max_len:
                    pad_width = self.max_len - input_seq.shape[1]
                    input_seq = np.pad(input_seq, ((0, 0), (0, pad_width)), constant_values=0)

                #predict next token
                pred_pos = min(len(generated)-1, self.max_len - 1)
                preds = self.model.predict(input_seq, verbose=0)[0, pred_pos]
                preds = np.log(preds + 1e-8) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                
                next_token = np.random.choice(range(self.vocab_size), p=preds)

                if next_token == 0:
                    break  # stop if padding token is predicted
                generated.append(next_token)

            generated_sequences.append(generated)

        return generated_sequences
    




    