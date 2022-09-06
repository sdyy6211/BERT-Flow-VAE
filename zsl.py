def zsl_model(
    dataset_path,
    label_name,
    text_name,
    processing_batch_size,
    zsl_res_save_path,
    zsl_res_save_name,
    cut = False,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import pandas as pd
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from utils import multi_label_predict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli').to(device)

    tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

    print('zero-shot learning model loaded')

    dataset = pd.read_csv(dataset_path,index_col=0)

    print('data loaded')

    total_num = np.ceil(dataset.shape[0]/processing_batch_size).astype(int)

    pred_list = []

    progress_bar = tqdm(range(0,total_num))

    if cut != False:

        datatext = [' '.join(i.split()[:cut]) for i in dataset.loc[:,text_name].values.tolist()]

    else:

        datatext = dataset.loc[:,text_name].values.tolist()

    print('processing data')

    for i in range(0,total_num):
        
        lower = i*processing_batch_size

        upper = (i+1)*processing_batch_size
        
        pred_bart = multi_label_predict(nli_model,tokenizer,datatext[lower:upper],label_name)
        
        pred_list.append(pred_bart)
        
        progress_bar.update(1)

    pred = torch.cat(pred_list).numpy()

    np.save('{0}/{1}'.format(zsl_res_save_path,zsl_res_save_name),pred)

    print('results saved')