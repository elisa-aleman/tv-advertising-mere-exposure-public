#-*- coding: utf-8 -*-

# import sqlite3
import csv
import os.path
import gensim
import codecs
import sklearn
from sklearn import svm
import numpy
from minepy import MINE

###SQL not using it
# # Just saving time connecting
# def Connect(sqlite_file):
#     conn = sqlite3.connect(sqlite_file)
#     c = conn.cursor()
#     return conn, c

def getProductList():
    tvbansuu_csv_path = os.path.join(processed_data_folder_path, "TVBanSuu.csv")
    _,tvbansuu = readCSV(tvbansuu_csv_path)
    product_list = [int(i[0]) for i in tvbansuu if int(i[1])>0]
    return product_list

def getXtitles():
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Heijitsu','Weekend','Holidays','Rest_Day']
    prime_statuses = ['Prime','NonPrime']
    x_titles = ["{}_{}".format(day,ps) for day in days for ps in prime_statuses]+days+prime_statuses+['Total']
    return x_titles

# type = 'total_only' ## 1D, only total amount of seconds
# type = 'day_prime'  ## Each day of the week prime/nonprime ###IGNORE DAY only (without prime/nonprime factor), IGNORE PRIME only (without day factor)
# type = 'combined'   ## day_prime + total_only
def getXvector_product(product_id, mltype='day_prime'):
    x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
    if mltype=='total_only':
        x = [[int(row[37])] for row in pre_x if int(row[1])==product_id]
    elif mltype=='day_prime':
        x = [[int(val) for val in row][2:24] for row in pre_x if int(row[1])==product_id]
    elif mltype=='combined':
        x = [[int(val) for val in row][2:24]+[int(row[37])] for row in pre_x if int(row[1])==product_id]
    return x

def getXvector_user(user_id, mltype='day_prime'):
    x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
    if mltype=='total_only':
        x = [[int(row[37])] for row in pre_x if row[0]==user_id]
    elif mltype=='day_prime':
        x = [[int(val) for val in row][2:24] for row in pre_x if row[0]==user_id]
    elif mltype=='combined':
        x = [[int(val) for val in row][2:24]+[int(row[37])] for row in pre_x if row[0]==user_id]
    return x

########

############################
########## Models ##########
############################

# corpus = vector >> using each title and its kinds of answers as different dimensions or "words"
# num_topics is the number of topics
# id2word = titles >> the column titles_answer are the "words" in our data
def LDA(vectorized, num_topics, vec_titles):
    titles = gensim.corpora.Dictionary(vec_titles)
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    lda = gensim.models.ldamodel.LdaModel(corpus=vector, num_topics=num_topics, id2word=titles)
    return lda

def HDP(vectorized, vec_titles):
    titles = gensim.corpora.Dictionary(vec_titles)
    # vector = [[(key,int(val)) if val!=' ' else (key,0) for key,val in enumerate(row)] for row in vector]
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    hdp = gensim.models.hdpmodel.HdpModel(corpus=vector, id2word=titles)
    return hdp

def tSNE(input_filename, output_filename, header=True, n_dim=2):
    if header:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0), skiprows=1)
    else:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0))
    compressed_data = sklearn.manifold.TSNE(n_dim).fit_transform(raw_data)
    numpy.savetxt(output_filename, compressed_data, delimiter=",")


def getMIC(x, y):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic

########