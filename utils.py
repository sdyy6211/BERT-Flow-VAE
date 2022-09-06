import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pos_mask(documents:List[str],
             tagger,
             tokenizer,
             id2label:Dict[int,str],
             keep_list:List[str],
             processing_batch_size:int) -> List[List[str]]:
    
    """
    pos masking tokens
    tagger: pos model
    tokenizer: tokenizer
    """
    
    pernum = processing_batch_size

    total_num = np.ceil(len(documents)/pernum).astype(int)

    processed_tokens = []

    attention_mask_list = []

    progress_bar = tqdm(range(total_num))
    
    for i in range(total_num):
        
        
        input_ids = tokenizer(documents[i*pernum:(i+1)*pernum],
                        padding=True,
                            truncation = True,
                        return_tensors='pt')['input_ids'].to(device)

        attention_mask = tokenizer(documents[i*pernum:(i+1)*pernum],
                                        padding=True,
                                        truncation = True,
                                        return_tensors = 'pt')['attention_mask'].to(device)

        masked_tokens = filter_out_words(input_ids,
                                    attention_mask,
                                    tagger,
                                    id2label,
                                    keep_list)

        processed_tokens.append(masked_tokens)

        attention_mask_list.append(attention_mask.detach().cpu().numpy())
        
        progress_bar.update(1)
    
    return post_padding(processed_tokens)

def tokens_to_documents(processed_tokens:List[List[str]],tokenizer) -> List[str]:

    """
    convert tokenized sentences in List back to String
    tokenizer: tokenizer
    """

    documents = []

    max_num = len(processed_tokens)

    progress_bar = tqdm(range(max_num))

    for ind,i in enumerate(processed_tokens):

        nonzero_words = tokenizer.convert_ids_to_tokens(i)

        filtered_document = [i for i in reverse_subword_tokenization(nonzero_words) if i != '']

        documents.append(' '.join(filtered_document))
        
        progress_bar.update(1)
        
    return documents

def reverse_subword_tokenization(data:List[str]) -> List[str]:

    """
    reverse subword tokenization and remove [PAD]
    """

    k_ = []

    for ind,i in enumerate(data):

        if i.startswith('##'):

            k_[-1] = k_[-1]+i.replace('##','')

        else:

            if i == '[PAD]':

                k_.append(i.replace('[PAD]',''))

            else:

                k_.append(i)

    return k_

def post_padding(list_:List[List["torch.Tensor"]]) -> List["torch.Tensor"]:

    """
    merge different input_inds into one
    """
    
    total_length = sum([i.shape[0] for i in list_])
    
    max_padding = max([i.shape[1] for i in list_])
    
    new_padded = np.zeros((total_length,max_padding))
    
    start_x_index = 0
    
    start_y_index = 0 # not used
    
    for i in range(len(list_)):
        
        end_x_index=start_x_index+list_[i].shape[0]
        
        end_y_index=start_y_index+list_[i].shape[1] # not used
        
        new_padded[start_x_index:end_x_index,start_y_index:end_y_index] = list_[i]
        
        start_x_index = end_x_index
        
    return new_padded

def filter_out_words(input_ids:"torch.Tensor",
                        attention_mask:"torch.Tensor",
                        postagger,
                        id2label: Dict[int,str],
                        keep_list:List[str]) -> "torch.Tensor":
    """
    pos processing
    postagger: pos model
    """
    
    label2id = {j:i for i,j in id2label.items()}
    
    with torch.no_grad():
        
        res = postagger(input_ids,attention_mask = attention_mask)[0].argmax(axis = 2).detach().cpu().numpy()
        
    posmask = np.zeros((res.shape))
        
    for i in keep_list:
        
        posmask += (res == label2id[i]) # only keep words in the keeplist
        
    return (input_ids.detach().cpu().numpy()*attention_mask.detach().cpu().numpy()*posmask).astype(int)

def custom_tokenizer(words:str):
    """
    helper function
    """

    return reverse_subword_tokenization(tokenizer.tokenize(words))

def train_tfidf(input_ids_list: List["torch.Tensor"], tokenizer):

    """
    train a sklearn tf-idf model
    tokenizer: tokenizer
    """
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    tokenized = [[str(w) for w in e[i].tolist()] for b in input_ids_list for e in b for i in torch.nonzero(e,as_tuple = True)]

    cusvocab = dict(zip([str(i) for i in range(tokenizer.vocab_size)],range(tokenizer.vocab_size)))

    tfidf = TfidfVectorizer(tokenizer=identity_token,lowercase=False,vocabulary  = cusvocab,norm = 'l1',token_pattern = None)
    
    tfidf.fit(tokenized)
    
    return tfidf,tokenized

def identity_token(x):

    """
    helper function
    """

    return x

def produce_tfidf_scores(tfidf, input_ids:"torch.Tensor", attention_masks:"torch.Tensor", alpha:int = 0.1) -> "torch.Tensor":

    """
    produce tfidf scores for input_ids to perform tf-idf weighted average
    tfidf: sklearn tfidf model
    """
    
    input_ids = (input_ids*attention_masks).cpu()
    
    weighed_attention_mask = torch.zeros_like(input_ids,dtype = torch.float,device = 'cpu')
    
    for idx in range(input_ids.shape[0]):
        
        ids = input_ids[idx,:]
        
        nonzeros = torch.nonzero(ids,as_tuple = True)[0]
        
        if len(nonzeros) > 0:
        
            tokens = ids[nonzeros].tolist()

            tokenized = [str(w) for w in tokens]

            tfidf_res = tfidf.transform([tokenized])

            weights = tfidf_res[0,tokens].todense()

            weights = weights/weights.sum()

            weights = weights*(1-alpha)+np.full(len(nonzeros),1/len(nonzeros))*alpha

            weighed_attention_mask[idx,nonzeros] = torch.tensor(weights,dtype = torch.float,device = 'cpu')
            
        else:
            
            weighed_attention_mask[idx,nonzeros] = input_ids[idx,nonzeros].to(torch.float)

    return weighed_attention_mask

def tfidf_mean_pooling(embeddings: "torch.Tensor",attention_mask: "torch.Tensor") -> "torch.Tensor":

    """
    perform tf-idf mean pooling to convert (batch,seq_len,embed_dim) into (batch,embed_dim)
    here the attention_mask is a weightest attention_mask by tf-idf scores
    """
    
    attention_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1).expand(embeddings.size()).float()
    
    return torch.sum(embeddings * attention_mask_expanded, 1)

def mean_pooling(embeddings:"torch.Tensor",attention_mask:"torch.Tensor") -> "torch.Tensor":

    """
    vanilla mean pooling to convert (batch,seq_len,embed_dim) into (batch,embed_dim)
    """
    
    attention_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1).expand(embeddings.size()).float()
    
    return torch.sum(embeddings * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)

def processing_inputs(
    train_set:"pd.DataFrame",
    text_name:str,
    processing_batch_size:int,
    tokenizer
)-> Tuple["torch.Tensor","torch.Tensor",List]:

    """
    processing inputs into (input_ids,attention_masks,original_index)
    original index can be used in the future to track the data
    tokenizer:tokenizer
    """

    N = train_set.shape[0]

    total_num = np.ceil(N/processing_batch_size).astype(int)

    input_ids_list = []

    attention_mask_list = []

    embedding_avgbert_list = []

    index_list = []

    progress_bar = tqdm(range(total_num))

    for i in range(total_num):

        lower = i*processing_batch_size

        upper = (i+1)*processing_batch_size-1

        with torch.no_grad():

            tokenized = tokenizer(train_set.loc[lower:upper,text_name].values.tolist(),
                                return_tensors='pt',padding=True,truncation = True)

            input_ids = tokenized['input_ids'].to(device)

            attention_mask = tokenized['attention_mask'].to(device)

        index_list += train_set.loc[lower:upper].index.values.tolist()

        input_ids_list.append(input_ids.detach().cpu())

        attention_mask_list.append(attention_mask.detach().cpu())

        progress_bar.update(1)
        
    return (input_ids_list,attention_mask_list,index_list)

def producing_embeddings(
    tfidf,
    input_ids_list:List["torch.Tensor"],
    attention_mask_list:List["torch.Tensor"],
    pooling:str,
    embedding_model
) -> "torch.Tensor":
    """
    produce embeddings of all layers using the language model
    tfidf: sklearn tf-idf model
    pooling: poolings strategy
    embedding_model: language model
    """
    embedding_avgbert_list = []

    for ids,masks in zip(input_ids_list,attention_mask_list):
        
        with torch.no_grad():

            embedding_avgbert = embedding_model(ids.to(device),masks.to(device))
        
        if pooling == 'tfidf':
            
            weighted_attention_mask = produce_tfidf_scores(tfidf,ids,masks)
            
            weighted_embedding = tfidf_mean_pooling(
                torch.stack(embedding_avgbert).permute(1,2,0,3),weighted_attention_mask.to(device)
            )
            
        elif pooling == 'average':
            
            weighted_embedding = mean_pooling(
                torch.stack(embedding_avgbert).permute(1,2,0,3),masks.to(device)
            )
            
        elif pooling == 'cls':
            
            weighted_embedding = torch.stack(embedding_avgbert).permute(1,2,0,3)[:,0,:,:]

        embedding_avgbert_list.append(weighted_embedding.cpu())
        
    return torch.cat(embedding_avgbert_list,axis = 0).view(-1,7,12,64) # reshape into (batch, layer, heads, dim_per_head)


def multi_label_predict(model,tokenizer,data:List[str],labels:List[str])->"torch.Tensor":

    """
    perform zero-shot topic classification using the prepared template
    model:0shottc model
    tokenizer:tokenizer
    """
    
    preds = []
    
    for l in labels:
        
        hypothesis = f'This example is about {l}.'
        
        x = tokenizer.batch_encode_plus([(i,hypothesis) for i in data],padding = True,truncation = True,return_tensors='pt')
        
        with torch.no_grad():
            
            logits = model(**x.to(device))[0]

        entail_contradiction_logits = logits[:,[0,2]]
        
        probs = entail_contradiction_logits.softmax(dim=1)
        
        prob_label_is_true = probs[:,1]
        
        preds.append(prob_label_is_true.detach().cpu())
        
    return torch.stack(preds,axis = 1)

