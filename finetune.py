def finetune_bert(
    dataset_path,
    text_name,
    model_save_path,
    model_save_name,
    num_epoches = 5,
    ):

    import pandas as pd
    from tqdm.auto import tqdm
    import torch
    from transformers import (
        AdamW,
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        get_scheduler,
    )
    from transformers import AutoTokenizer
    from datasets import Dataset,DatasetDict
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    from accelerate import Accelerator


    def tokenize_function(examples):
          
        tokenized = tokenizer_sbert(examples[text_name], return_special_tokens_mask=True,padding=True,truncation=True)
        
        return tokenized


    tokenizer_sbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    comments = pd.read_csv(dataset_path,index_col=0).loc[:,text_name]

    print('dataset loaded')

    finetune_train = pd.DataFrame(comments.values,columns=[text_name])

    dataset = DatasetDict({'train':Dataset.from_pandas(finetune_train)})

    print('tokenizing')

    tokenized_datasets = dataset.map(tokenize_function,
                                    batched=True,
                                    remove_columns=[text_name])

    print('tokenizing finished')

    train_dataset = tokenized_datasets["train"]

    print('dataset pre-processing finished')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_sbert, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')

    print('pre-trained model loaded')

    accelerator = Accelerator()

    device = accelerator.device

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=32)

    no_decay = ["bias", "LayerNorm.weight"]

    parameter_groups_high_nodecay = {'params':[],
                                     'lr':1e-3,
                                    'weight_decay':0.0}

    parameter_groups_high_decay = {'params':[],
                            'lr':1e-3,
                            'weight_decay':1e-3}

    parameter_groups_low_nodecay = {'params':[],
                            'lr':1e-5,
                            'weight_decay':0.0}

    parameter_groups_low_decay = {'params':[],
                            'lr':1e-5,
                            'weight_decay':1e-3}

    for name,p in model.distilbert.named_parameters():
        
        if 'transformer.layer' in name:
            
            if int(name.split('.')[2]) >= 3:
                
                if any(nd in name for nd in no_decay):
                
                    parameter_groups_high_nodecay['params'].append(p)
                    
                else:
                    
                    parameter_groups_high_decay['params'].append(p)
                
            else:
                
                if any(nd in name for nd in no_decay):
                
                    parameter_groups_low_nodecay['params'].append(p)
                    
                else:
                    
                    parameter_groups_low_decay['params'].append(p)
                
        else:
            
            if any(nd in name for nd in no_decay):
            
                parameter_groups_low_nodecay['params'].append(p)
                
            else:
                
                parameter_groups_low_decay['params'].append(p)

    optimizer_grouped_parameters = [parameter_groups_high_nodecay,
                                    parameter_groups_high_decay,
                                    parameter_groups_low_nodecay,
                                    parameter_groups_low_decay]

    optimizer = AdamW(optimizer_grouped_parameters)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    num_train_epochs = num_epoches

    max_train_steps = len(train_dataloader)*num_train_epochs

    lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps*0.1),
            num_training_steps=max_train_steps,
        )

    print('start to finetune the model')

    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(num_train_epochs):
          
        model.train()
        
        for batch in train_dataloader:
              
            optimizer.zero_grad()
            
            outputs = model(**batch)
            
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            optimizer.step()
            
            lr_scheduler.step()
            
            progress_bar.update(1)
            
    print('finetuning finished')

    torch.save(model.distilbert.state_dict(),'{0}/{1}'.format(model_save_path,model_save_name))

    print('model saved')
