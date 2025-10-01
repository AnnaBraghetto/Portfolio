from itertools import product

from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

class SequencePreprocess():

    def __init__(self, sequence, label=None):
        '''
        Initialization of the SequencePreprocess class.
        Input:
        sequence =  pd.dataframe, nucleotide sequences
        Output:
        '''

        # store the sequence
        self.sequence = sequence

    
    def tokenize(self,col_name,k=6):
        '''
        Tokenize the nucleotide sequences into individual characters.
        Input:
        col_name = string, column name for nucleotide sequences
        k = int, length of each k-mer
        Output:
        '''

        #k-mer
        def generate_kmers(sequence, k=6):
            '''
            Generate k-mers from a nucleotide sequence
            Input:
            sequence = string, e.g. "ATGCGT"
            k = int, length of each k-mer
            Output: list of k-mers, e.g. ["ATGCGT", "TGCGTA", "GCGTAC"]
            '''

            return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

        def generate_kmers_batch(sequences, k=6):
            '''
            Generate k-mers for a list of sequences
            Input:
            sequences =  list of strings, e.g. ["ATGCGT", "GGCTAAGT"]
            Output: list of lists of k-mers
            '''

            return [generate_kmers(seq, k) for seq in sequences]
        
        #save k
        self.k = k
        
        #get tokens
        self.tokens = generate_kmers_batch(self.sequence[col_name], k=k)


    def encode(self):
        '''
        Encode the nucleotide sequences
        Input:
        Output:
        '''

        def kmer_seq_to_ids(seq, kmer2id):
            '''
            Convert a k-mer sequence to a list of ids
            Input:
            seq = list of k-mers, e.g. ["ATGCGT", "TGCGTA", "GCGTAC"]
            kmer2id = dictionary mapping k-mers to ids
            Output: list of ids, e.g. [1, 2, 3]
            '''

            #return list of ids
            return [kmer2id[k] for k in seq]
        

        nucleotides = ['A','C','G','T']
        all_kmers = [''.join(p) for p in product(nucleotides, repeat=self.k)]

        #map k-mer to id (0 reserved for padding)
        kmer2id = {kmer: idx+1 for idx, kmer in enumerate(all_kmers)}

        self.encoding = [kmer_seq_to_ids(seq, kmer2id) for seq in self.tokens]


    def pad_sequences(self, max_len):
        '''
        Pad the encoded sequences to a uniform length
        Input:
        maxlen = int, maximum length to pad/truncate sequences to
        Output:
        '''

          
        self.padded = pad_sequences(self.encoding, maxlen=max_len, padding='post', truncating='post')
        

        
    

    