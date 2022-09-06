def tf_idf_processing(
    dataset_path,
    text_name,
    processing_batch_size,
    tfidf_save_path,
    tfidf_save_name,
):

    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    import pickle
    from utils import (train_tfidf, 
                    identity_token, 
                    processing_inputs
                    ) 

    tokenizer_sbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_set = pd.read_csv(dataset_path,index_col=0)

    print('dataset loaded')

    print('tokenizing data')

    input_ids_list,attention_mask_list,index_list = processing_inputs(train_set = train_set,
                                                                    text_name = text_name,
                                                                    processing_batch_size = processing_batch_size,
                                                                    tokenizer = tokenizer_sbert)

    print('processing tfidf data')

    tfidf,tokenized = train_tfidf(input_ids_list, tokenizer_sbert)

    print('finished processing tfidf data')

    with open('{0}/{1}'.format(tfidf_save_path,tfidf_save_name),'wb') as f:
        pickle.dump(tfidf,f)

    print('tfidf model saved')