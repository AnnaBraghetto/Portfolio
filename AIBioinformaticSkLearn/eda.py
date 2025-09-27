import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df):
        '''Initialization of class EDA
        Input: 
        df = pandas dataframe containing data
        Output:
        '''

        #set amino name
        col = ['A', 'R', 'N', 'D', 'C', 
               'Q', 'E', 'G', 'H', 'I', 
               'L', 'K', 'M', 'F', 'P', 
               'S', 'T', 'W', 'Y', 'V','diabete'] 
        
        #check if columns have names
        if df.columns.isnull().any() or df.columns[0] == 0:
            #assign default names col_0, col_1, ...
            df.columns = [col[i] for i in range(df.shape[1])]
        
        #store a copy of the dataframe
        self.df = df.copy()
        
        #define target as the last column
        self.target = df.columns[-1]
        
        #define features as all columns except target
        self.features = df.columns[:-1]


    def overview(self):
        '''Method to get overview of df
        Input:
        Output:
        '''

        #print shape and basic info
        print('shape:', self.df.shape)
        print('\ndata types:\n', self.df.dtypes)
        print('\nfirst 5 rows:\n', self.df.head())
        print('\nsummary statistics:\n', self.df.describe())


    def missing_values(self):
        '''Method to check missing values
        Input:
        Output:
        '''

        #check for missing values
        missing = self.df.isnull().sum()
        missing = missing[missing>0]
        if missing.empty:
            print('no missing values found')
        else:
            print('missing values per column:\n', missing)
            sns.heatmap(self.df.isnull(), cbar=False)
            plt.title('missing values heatmap')
            plt.show()


    def histograms(self):
        '''Method to plot feature distributions
        Input:
        Output:
        '''

        #histograms of numeric features
        self.df[self.features].hist(figsize=(10,6), bins=20)
        plt.suptitle('feature distributions')
        plt.tight_layout()
        plt.show()


    def boxplots(self):
        '''Method to plot boxplot of features
        Input:
        Output:
        '''
        
        #boxplots of numeric features
        for col in self.features:
            if np.issubdtype(self.df[col].dtype, np.number):
                sns.boxplot(y=self.df[col])
                plt.title(f'boxplot of {col}')
                plt.show()


    def correlation_matrix(self):
        '''Method to plot correlation matrix of features
        Input:
        Output:
        '''
        
        #correlation matrix for numeric features
        corr = self.df[self.features].corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, annot=True,fmt='.2f', cmap='coolwarm')
        plt.title('correlation matrix')
        plt.show()


    def target_distribution(self):
        '''Method to plot target distribution
        Input:
        Output:
        '''

        #plot target distribution
        if np.issubdtype(self.df[self.target].dtype, np.number):
            sns.histplot(self.df[self.target], kde=True)
        else:
            sns.countplot(x=self.df[self.target])
        plt.title(f'distribution of target: {self.target}')
        plt.show()