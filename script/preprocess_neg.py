from sqlalchemy import create_engine
import pymysql
import pandas as pd
import pickle
import torch
import json
from time import time
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util,LoggingHandler
import logging
from utils import Config
# logging.basicConfig(format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO)
config = Config(config_file='/data/yuchen/projects/sentence_embedding/config/process_neg.json')
model = SentenceTransformer(config['model_path'])


pos_pair = pickle.load(open(config['pos_pair_path'],'rb'))
lni_content = pos_pair[['lni','content']].dropna().groupby('lni')
lni_content = lni_content.agg(lambda x: list(x.unique()))

lnis = pickle.load(open(config['lnis_path'],'rb'))
# pos_train_date_2 = pickle.load(open('/data/yuchen/projects/sentence_embedding/data/train/pos_train_date_2.pkl','rb'))
# pos_train_date_3 = pickle.load(open('/data/yuchen/projects/sentence_embedding/data/train/pos_train_date_3.pkl','rb'))

'init mysql connection'
db_connection_str = 'mysql+pymysql://shenjiawei:jiaweiDH$z048Kue2*34@cat-cluster.cluster-cvieeiq0uwtk.ap-southeast-1.rds.amazonaws.com:3306/ai_cat_ca_dev'
db_connection = create_engine(db_connection_str)

def get_lni_sentences(lni_content,start,end):
    keyword_sql= "select l.lni, c.content from lni_unique as l inner join case_sentence as c on l.lni = c.lni where l.lni in " + '(' + ','.join(list(map(lambda x:'"' + x + '"',lni_content.index[start:end]))) + ')'
    result=pd.read_sql(keyword_sql, con=db_connection)
    lni_sentence = result.groupby('lni').agg(list)
    return lni_sentence

def transformer_rank(gold_sents,all_sents,topk):
    #Compute embedding for both lists
    embeddings1 = model.encode(gold_sents, convert_to_tensor=True)
    embeddings2 = model.encode(all_sents, convert_to_tensor=True)
    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    result = defaultdict(list)
    try:
        if config['neg_pos'] =='neg':
            tensor, indice = torch.topk(cosine_scores,topk,largest=False)
        else:
            tensor, indice = torch.topk(cosine_scores,topk,largest=True)
    except Exception:
        print('selected k out of range')
        return None
    unsimilar_sents = []
    for i in range(indice.shape[0]):
        # print("gold_sent{}----------------------------------------------------------------".format(i))
        # print(gold_sents[i])
        for j in range(indice.shape[1]):
            result[gold_sents[i]].append((all_sents[indice[i,j]],str(cosine_scores[i][indice[i,j]].detach().numpy()))) 
            unsimilar_sents.append(all_sents[indice[i,j]])
        #     print("\n{} \n Score: {:.4f}".format(all_sents[indice[i,j]], cosine_scores[i][indice[i,j]]))
        # print("--------------------------------------------------------------------------")
    return unsimilar_sents

def get_neg_pair(lni_content,lni_sentance,topk):
    neg_train_data = pd.DataFrame(columns=list(POS_TRAIN_DATA))
    for index in lni_sentance.index:
        # index = '5F16-93C1-DY89-M2CC-00000-00'
        paths = POS_TRAIN_DATA[POS_TRAIN_DATA.lni == index].path.unique()
        unrelevant_sents = transformer_rank(lni_content.content[index],lni_sentance.content[index],topk)
        rows = defaultdict(list)
        if unrelevant_sents:
            unrelevant_sents = set(unrelevant_sents)
            for p in paths:
                for s in unrelevant_sents:
                    rows['lni'].append(index)
                    rows['path'].append(p)
                    rows['content'].append(s)
                    if config['neg_pos'] =='neg':
                        rows['lable'].append(0)
                    else:rows['lable'].append(1)
            neg_train_data = neg_train_data.append(pd.DataFrame(rows),ignore_index=True)
        else:
            print('problematic index: {}'.format(index)) 
            # neg_train_data = None
            continue
    return neg_train_data

step = config['step']
POS_TRAIN_DATA =  pickle.load(open(config['pos_pair_pathcut_path'],'rb'))
POS_TRAIN_DATA.drop_duplicates(inplace=True)
print('after dropping duplicated rows, remaining pos_train_date shape is {}'.format(POS_TRAIN_DATA.shape[0]))
THRESHHOLD = config['threshhold']
# step = 10
# POS_TRAIN_DATA =  pos_train_date_2
# THRESHHOLD = 10

start = 0
end = step
neg_train_data = pd.DataFrame(columns=list(POS_TRAIN_DATA))

start_time = time()
for epoch in range(len(lni_content.index)//100):
    echo_start_time = time()
    try:
        print('fetching data from mysql')
        lni_sentance = get_lni_sentences(lni_content,start,end)
        print('fetching data finished')
    except Exception:
        continue
    neg_pair = get_neg_pair(lni_content,lni_sentance,2)
    start = start + step
    end = end + step
    if neg_pair is None: 
        continue
    else:
        neg_train_data = neg_train_data.append(neg_pair,ignore_index=True)
    print('{} epochs is finished, {} samples are created, cost {} s'.format(epoch+1,neg_train_data.shape[0],time()-echo_start_time))
    if neg_train_data.shape[0] > THRESHHOLD: 
        print('{} samples are created'.format(neg_train_data.shape[0]))
        # print('{} negative samples are created'.format(neg_train_data.shape[0]))
        break
neg_train_data.to_pickle(config['out_put_path'])
print('train_data is stored, cost {} s'.format(time() - start_time))

