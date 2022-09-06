def produce_embeddings(
      dataset_path,
      model_path,
      text_name,
      processing_batch_size,
      embedding_save_path,
      embedding_save_name,
      pooling,
      tf_idf_model_path
      ):

      import torch
      import pandas as pd
      import pickle
      from transformers import AutoTokenizer, AutoModel
      from models import AVGBERT
      from utils import (
                         processing_inputs,
                         producing_embeddings
                         ) 

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      tokenizer_sbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')

      bert = AutoModel.from_pretrained('distilbert-base-uncased').to(device)

      bert.load_state_dict(torch.load(model_path))

      for p in bert.parameters():
            
          p.requires_grad = False

      avgbert = AVGBERT(bert)

      print('model loaded')

      train_set = pd.read_csv(dataset_path,index_col=0)

      print('dataset loaded')

      print('tokenizing data')

      input_ids_list,attention_mask_list,index_list = processing_inputs(train_set = train_set,
                                                                        text_name = text_name,
                                                                        processing_batch_size = processing_batch_size,
                                                                        tokenizer = tokenizer_sbert)

      print('loading tfidf data')

      with open(tf_idf_model_path,'rb') as f:

        tfidf = pickle.load(f)

      print('finished loading tfidf data')

      print('producing embeddings')

      avgbert_embedding = dict(
          zip(
              index_list,
              producing_embeddings(
                    tfidf,
                    input_ids_list,
                    attention_mask_list,
                    pooling,
                    avgbert)
          )
      )

      torch.save(avgbert_embedding,'{0}/{1}'.format(embedding_save_path,embedding_save_name))

      print('embeddings saved')
