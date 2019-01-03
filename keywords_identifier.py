import pandas as pd
import numpy as np
from nltk.corpus import stopwords

np.random.seed(1)
pd.options.display.max_colwidth = 800 # otherwise project descriptions are cut-off (don't want them too long either)

# import CRS-data
df_2017 = pd.read_csv("CRS_2017_181229.csv", header = 0)
df_2016 = pd.read_csv("CRS_2016_181229.csv", header = 0)
df_2015 = pd.read_csv("CRS_2015_181229.csv", header = 0)
df = pd.concat([df_2017, df_2016, df_2015])

# replace NA with zero
df = df.fillna(0)

# take a smaller subsample to speed up testing and debugging
#msk = np.random.rand(len(df)) < 0.01
#df = df[msk]
#print(len(df))

# Misc information about CRS data
headers = list(df)
table = pd.pivot_table(df, values='Value', index = ['YEAR'], aggfunc=np.sum).reset_index().rename_axis(None, axis = 1)
n = len(df.index)
print("Columns", headers)
print("Total number of contributions", n)
print("Total outcome per year", table)

# the policy objective we want to study
objective = 'Environment'

# get project descriptions for contributions whose disbursements > 0 and policy objective marker = 1, 2
df_12 = df.loc[df[objective].isin([1, 2]) & df['Value'] != 0][['Long Description']]
n_12 = len(df_12.index)

# get project descriptions for contributions whose disbursements > 0 and policy objective marker = 0
df_0 = df.loc[df[objective].isin([0]) & df['Value'] != 0][['Long Description']]
n_0 = len(df_0.index)

# split data into training (95%) and test samples (5%)
msk_12 = np.random.rand(n_12) < 0.95
df_train_12 = df_12[msk_12]
df_test_12 = df_12[~msk_12]
n_train_12 = len(df_train_12.index)
n_test_12 = n_12 - n_train_12
msk_0 = np.random.rand(n_0) < 0.95
df_train_0 = df_0[msk_0]
df_test_0 = df_0[~msk_0]
n_train_0 = len(df_train_0.index)
n_test_0 = n_0 - n_train_0

print("n_train_12", n_train_12)
print("n_test_12", n_test_12)
print("n_train_0", n_train_0)
print("n_test_0", n_test_0)

# import stop words to remove when comparing project descriptions
stopwords_all = set()
for i in ['english', 'spanish', 'french', 'german']:
    stopwords_all = stopwords_all.union(set(stopwords.words(i)))
stopwords_all = list(stopwords_all)

## Find key words to separate 1, 2 from 0 ##
# storage vector and parameters
key_words = [] # word to separate 1, 2 from 0 will be added to this storage vector
k_12 = 2 # number of contributions from 1, 2 to sample per comparison
k_0 = 25 # number of contributions from 0 to sample per comparison #
count = 0 # keep track of number of times samples that are compared
n_key_words = 800 # number of key words we want to have likelihood for and use for classification

while len(key_words) < n_key_words:
    sample_12 = [] #reset sample
    sample_0 = []

    # convert pandas objects to list of unique words and add to sample (chosen randomly)
    check = []  # used to check that duplicate descriptions aren't sampled
    for i in range(k_12):
        rand = list(set(df_train_12.iloc[[np.random.randint(0, n_train_12)]].to_string().split()))
        while len(rand) in check:
            rand = list(set(df_train_12.iloc[[np.random.randint(0, n_train_12)]].to_string().split()))
        check.append(len(rand))
        sample_12.append(rand)
    check = []
    for i in range(k_0):
        rand = list(set(df_train_0.iloc[[np.random.randint(0, n_train_0)]].to_string().split()))
        while len(rand) in check:
            rand = list(set(df_train_0.iloc[[np.random.randint(0, n_train_0)]].to_string().split()))
        check.append(len(rand))
        sample_0.append(rand)

    # make into lower cases and remove ',','.' etcetera
    for i in range(len(sample_12)):
        sample_12[i] = [x.lower() for x in sample_12[i]]
        for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
            sample_12[i] = [x.replace(j, '') for x in sample_12[i]]
    for i in range(len(sample_0)):
        sample_0[i] = [x.lower() for x in sample_0[i]]
        for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
            sample_0[i] = [x.replace(j, '') for x in sample_0[i]]

    # find common words in 1, 2 sample
    common_12 = []
    for i in range(1, len(sample_12)):
        if i == 1:
            common_12 = list(set(sample_12[i-1]).intersection(set(sample_12[i])))
        else:
            common_12 = list(set(common_12).intersection(set(sample_12[i])))

    # remove words which is also in 0 sample
    for i in sample_0:
        common_12 = [x for x in common_12 if (x not in i)]

    # remove stop words or integers or empty strings
    common_12 = [x for x in common_12 if x not in stopwords_all]
    try:
        common_12 = [x for x in common_12 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    except IndexError:
        pass
    common_12 = [x for x in common_12 if x not in ['']]

    # add new keywords
    key_words = key_words + list(set(common_12) - set(key_words))

    # cut off key words if too many words are added during last loop
    if len(key_words) > n_key_words:
        key_words = key_words[:n_key_words]

    count += 1
print(key_words)

print("Comparisons done before finding key words:", count)
np.savetxt('key_words.txt', np.array(key_words), fmt='%s')

## Detect presence of keywords ##

X = np.zeros((n_train_12+n_train_0, n_key_words)) # will contain input train data neural network
Y = np.zeros((n_train_12+n_train_0, 1)) # will contain output train neural network
for i in range(n_train_12):
    x_i = np.ones((1, len(key_words)))
    a = set(df_train_12.iloc[[i]].to_string().split()) # project description i:th item in this group
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        x_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    X[i,:] = x_i[0]
    Y[i,:] = 1 # we use 1 to indicate policy objective one or two and 0 to indicate policy objective 0

for i in range(n_train_0):
    x_i = np.ones((1, len(key_words)))
    a = set(df_train_0.iloc[[i]].to_string().split()) # project description i:th item in this group
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        x_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    X[n_train_12 + i, :] = x_i[0] # note index
    # no need to use Y[i,:] = 0 here because array initialized with zeros

# save array so that it isn't necessary to identify all keywords each time a new neural network etc. is trained
np.savetxt('X_train.txt', X, fmt='%d')
np.savetxt('Y_train.txt', Y, fmt='%d')

X = np.zeros((n_test_12+n_test_0, n_key_words)) # will contain input test neural network
Y = np.zeros((n_test_12+n_test_0, 1)) # will contain output test data neural network
for i in range(n_test_12):
    x_i = np.ones((1, len(key_words)))
    a = set(df_test_12.iloc[[i]].to_string().split()) # project description i:th item in this group
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        x_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    X[i,:] = x_i[0]
    Y[i,:] = 1 # we use 1 to indicate policy objective one or two and 0 to indicate policy objective 0

for i in range(n_test_0):
    x_i = np.ones((1, len(key_words)))
    a = set(df_test_0.iloc[[i]].to_string().split()) # project description i:th item in this group
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        x_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    X[n_test_12 + i, :] = x_i[0] # note index
    # no need to use Y[i,:] = 0 here because array initialized with zeros

# save array so that it isn't necessary to identify all keywords each time a new neural network etc. is trained
np.savetxt('X_test.txt', X, fmt='%d')
np.savetxt('Y_test.txt', Y, fmt='%d')