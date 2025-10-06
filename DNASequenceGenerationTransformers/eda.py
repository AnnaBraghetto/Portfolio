import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class SequenceEDA:
    def __init__(self, dataframe, seq_col='sequence', label_col='gene_type'):
        '''
        Initialization of the SequenceEDA class.
        Input:
        dataframe = pd.DataFrame containing nucleotide sequences and their labels
        seq_col = column name for nucleotide sequences
        label_col = column name for gene type labels
        Output:
        '''

        #copy input dataframe to avoid modifying original data
        self.df = dataframe.copy()
        #save seq_col
        self.seq_col = seq_col
        #save label_col
        self.label_col = label_col

        #add a column with the length of each nucleotide sequence
        self.df['length'] = self.df[self.seq_col].str.len()
    

    def plot_gene_type_distribution(self):
        '''
        Plot the distribution of gene types (labels)
        Input:
        Output:
        '''

        #prepare figure size
        plt.figure(figsize=(10, 5))
        #plot countplot
        sns.countplot(
            x=self.label_col, 
            data=self.df, 
            order=self.df[self.label_col].value_counts().index
        )

        #set title and labels
        plt.title('Distribution of Gene Types')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    
    def plot_sequence_length_distribution(self):
        '''
        Plot the distribution of sequence lengths, overall and per gene type
        Input:
        Output:
        '''

        #histogram of sequence lengths
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['length'], bins=30, kde=True)
        plt.title('Sequence Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

        #boxplot of lengths by gene type
        plt.figure(figsize=(12, 5))
        sns.boxplot(x=self.label_col, y='length', data=self.df)
        plt.title('Sequence Length by Gene Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
    def plot_nucleotide_frequencies(self):
        '''
        Plot the overall relative frequencies of A, T, G, C in all sequences
        Input:
        Output:
        '''

        #prepare to count nucleotides
        total_counts = Counter()
        total_bases = 0

        #count nucleotides across all sequences
        for seq in self.df[self.seq_col]:
            seq = seq.upper()
            counts = Counter(seq)
            total_counts.update(counts)
            total_bases += len(seq)
        
        #only consider canonical nucleotides
        nucleotides = ['A', 'T', 'G', 'C']
        freqs = {nt: total_counts[nt] / total_bases for nt in nucleotides}

        #bar plot of nucleotide frequencies
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(freqs.keys()), y=list(freqs.values()))
        plt.title('Global Nucleotide Frequencies')
        plt.ylabel('Relative Frequency')
        plt.ylim(0, 0.5)
        plt.tight_layout()
        plt.show()
