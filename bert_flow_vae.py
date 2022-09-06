def bert_flow_vae(
            embedding_path,
            topic_res_path,
            model_save_path,
            model_save_name,
            bert_flow_vae_pred_path,
            bert_flow_vae_pred_name,
            hps = True,
            num_epoches = 10,
            gamma = 1,
            training_batch_size = 512,
            BERT_DIM = 768,
            HIDDEN_DIM = 512,
            FC_DIM = 768,
            DIMENSION_SEQUENCE = [96,196,384,512,768],
            processing_batch_size = 1024,
            ):
      import torch
      from torch import nn
      import numpy as np
      from torch.utils.data import DataLoader,Dataset
      from torch.optim import AdamW
      from tqdm.auto import tqdm

      from models import Encoder,Decoder,VAEModel

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      class VAEDataset(Dataset):
            
          def __init__(self,embedding,topic_model_res):
                
              self.embedding = embedding
              
              self.topic_model_res = topic_model_res
              
          def __len__(self):
                
              return self.embedding.shape[0]
              
          def __getitem__(self,index):
                
              return self.embedding[index,:],self.topic_model_res[index,:]
            
      bertflow_embedding = torch.load(embedding_path)

      print('embeddings loaded')

      bertflow_embedding_values = torch.stack(list(bertflow_embedding.values()))

      if not bertflow_embedding_values.shape[1:]==torch.Size([64, 7, 12]):

            bertflow_embedding_values = bertflow_embedding_values.permute(0,3,1,2)

      bertflow_embedding_values = bertflow_embedding_values[:,:,[0,1,5],:].mean(axis = 2).reshape(-1,64*12)

      topic_model_res = np.load(topic_res_path)

      print('topic results loaded')

      LATENT_DIM = topic_model_res.shape[1]

      encoder = Encoder(
                     BERT_DIM,
                     HIDDEN_DIM,
                     FC_DIM,
                     LATENT_DIM,)

      decoder = Decoder(LATENT_DIM,
                        DIMENSION_SEQUENCE,
                        BERT_DIM)

      vae_model = VAEModel(encoder,decoder).to(device)

      vaedataset = VAEDataset(bertflow_embedding_values,topic_model_res)

      vaedataloader = DataLoader(vaedataset,batch_size = training_batch_size,
                                 shuffle = True,drop_last = True)

      factor = torch.tensor(LATENT_DIM/10,dtype = torch.float).to(device)

      _ = vae_model.train()

      total_epoch = num_epoches

      total_count = len(vaedataloader)*total_epoch

      progress_bar = tqdm(range(total_count))

      no_decay = ["bias", "LayerNorm.weight"]

      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in vae_model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": 1e-2,
          },
          {
              "params": [p for n, p in vae_model.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]

      optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)

      print('training the model')

      for epoch in range(1,total_epoch+1):
          
          for data_batch in vaedataloader:
              
              with torch.no_grad():

                  batchn = len(data_batch)

                  embedding = data_batch[0].to(torch.float).to(device)
                  
                  lda_res = data_batch[1].to(torch.float).to(device)

                  x = embedding
              
              optimizer.zero_grad()
              
              x_hat, mean_, log_var_ = vae_model(embedding)

              if hps:
              
                    alpha = np.sqrt(gamma)*0.001 if epoch < 2 else np.sqrt(gamma)*0.1

                    beta = gamma*factor if epoch < total_epoch else gamma*factor*0.5

              else:

                    alpha = np.sqrt(gamma)*0.1

                    beta = gamma*factor
              
              topic_loss = (nn.BCEWithLogitsLoss(reduction = 'sum')(mean_,lda_res)).div(LATENT_DIM*batchn)
              
              reproduction_loss = torch.nn.functional.mse_loss(x_hat, x,reduction='sum').div(BERT_DIM*batchn)
          
              KLD = (- 0.5 * torch.sum(1 + log_var_ - mean_.pow(2) - log_var_.exp())).div(LATENT_DIM*batchn)
              
              loss = reproduction_loss + alpha*KLD + beta*topic_loss
              
              loss.backward()
              
              optimizer.step()

              progress_bar.update(1)

      print('training finished')

      torch.save(vae_model.state_dict(), '{0}/{1}'.format(model_save_path,model_save_name))

      print('model saved')

      print('making predictions')

      _ = vae_model.eval()
      
      vae_model.random_ = False
      
      total_num = np.ceil(bertflow_embedding_values.shape[0]/processing_batch_size).astype(int)

      mean_list = []

      progress_bar = tqdm(range(total_num))

      for i in range(total_num):
          
          lower = i*processing_batch_size

          upper = (i+1)*processing_batch_size
          
          with torch.no_grad():
              
              embedding = bertflow_embedding_values[lower:upper,:].to(device)
              
              x = embedding.view(-1,12*64)
              
              x_hat, mean, log_var = vae_model(embedding)

          mean_list.append(mean)
          
          progress_bar.update(1)

      print('finished making predictions')

      embedding_bertflow_vae_mean = torch.cat(mean_list,axis = 0)

      mean_total = embedding_bertflow_vae_mean.detach().cpu().numpy()

      np.save('{0}/{1}'.format(bert_flow_vae_pred_path,bert_flow_vae_pred_name),mean_total)

      print('predictions saved')
