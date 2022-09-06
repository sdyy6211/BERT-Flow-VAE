def pos_processing(
      dataset_path,
      text_name,
      tokens_save_path,
      tokens_save_name,
      documents_save_path,
      documents_save_name,
      processing_batch_size = 1024,
      ):
      import torch
      import pandas as pd
      import numpy as np
      from tqdm.auto import tqdm
      from transformers import AutoTokenizer
      from transformers import AutoModelForTokenClassification, AutoTokenizer

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      from utils import pos_mask, tokens_to_documents

      train_set = pd.read_csv(dataset_path)

      print('dataset loaded')

      tokenizer_sbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')

      bertpos_name ="vblagoje/bert-english-uncased-finetuned-pos"

      bertpos = AutoModelForTokenClassification.from_pretrained(bertpos_name).to(device)

      for p in bertpos.parameters():
            
          p.requires_grad = False
          
      id2label =  {
            0: "ADJ",
          1: "ADP",
          2: "ADV",
          3: "AUX",
          4: "CCONJ",
          5: "DET",
          6: "INTJ",
          7: "NOUN",
          8: "NUM",
          9: "PART",
          10: "PRON",
          11: "PROPN",
          12: "PUNCT",
          13: "SCONJ",
          14: "SYM",
          15: "VERB",
          16: "X"
        }

      print('model loaded')

      print('producing masked tokens')

      maskedtokens = pos_mask(train_set.loc[:,text_name].values.tolist(), bertpos, tokenizer_sbert, id2label,['NOUN','ADJ'],processing_batch_size)

      print('finished masked tokens')

      np.save('{0}/{1}'.format(tokens_save_path, tokens_save_name),maskedtokens)

      print('masked tokens saved')

      print('converting masked tokens to documents')

      documents = tokens_to_documents(maskedtokens, tokenizer_sbert)

      print('finished converting')

      with open('{0}/{1}'.format(documents_save_path,documents_save_name),'w',encoding = 'utf-8') as f:
          f.write(str(documents))

      print('documents saved')
