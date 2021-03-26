import pandas as pd
import pickle
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util,LoggingHandler
model_save_path_2 = 'script/output/new-pos-samples-pos1.3million-neg1.3million-distilbert-base-nli-mean-tokens-2021-03-11_14-08-37'
model_save_path_3 = 'script/output/new-pos-samples-pos2million-neg2million-level3distilbert-base-nli-mean-tokens-2021-03-17_14-39-52'
models_finutuned = {2:SentenceTransformer(model_save_path_2),3:SentenceTransformer(model_save_path_3)}
quick_dis = pickle.load(open('data/model/path_map/quick_dis.bin','rb'))
quick_map = pickle.load(open('data/model/path_map/quick_map.bin','rb'))
quick_route = pickle.load(open('data/model/path_map/quick_route.bin','rb'))
quick_string_map = pickle.load(open('data/model/path_map/quick_string_map.bin','rb'))

def id2sen1(sen_id):
    sen1 = [quick_map[v] for v in quick_route[sen_id]]
    sen1.reverse()
    return sen1


def transformer_rank(sens1_ids,sens2,models,level=2,topk=1):
    #get sens1 via index
    sens1 = ['-'.join(id2sen1(index)[:level]) for index in sens1_ids]
    # select model : level 2 or level 3
    model = models[level]
    #Compute embedding for both lists
    embeddings1 = model.encode(sens1, convert_to_tensor=True)
    embeddings2 = model.encode(sens2, convert_to_tensor=True)
    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    result = defaultdict(list)
    tensor, indice = torch.topk(cosine_scores,topk,largest=True)
    similar_sents = defaultdict(list)
    for i in range(indice.shape[0]):
        for j in range(indice.shape[1]):
            result[sens1[i]].append((sens2[indice[i,j]],str(cosine_scores[i][indice[i,j]].detach().numpy()))) 
#             similar_sents.append(all_sents[indice[i,j]])
            similar_sents[sens1[i]].append(sens2[indice[i,j]])
    return similar_sents

if __name__ == '__main__':
    print(transformer_rank(['ABR1','ABR5'],['all','hello','fusion'],models_finutuned,level=2,topk=3))
    print(transformer_rank(['ABR1','ABR5'],['all','hello','fusion'],models_finutuned,level=3,topk=3))