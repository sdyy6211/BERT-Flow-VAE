def bert_flow(
      embedding_path,
      model_save_path,
      model_save_name,
      embedding_save_path,
      embedding_save_name,
      training_batch_size = 1024,
      num_epoches = 5,
      processing_batch_size = 1024,
      ):


      import torch
      from tqdm.auto import tqdm
      import numpy as np
      from torch.utils.data import DataLoader,Dataset

      from models import Glow

      class BERTFLOWDATASET(Dataset):
          
          def __init__(self,dataset):
              self.dataset = dataset
              
          def __len__(self):
              return self.dataset.shape[0]
              
          def __getitem__(self,index):
              return self.dataset[index,:]

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      avgbert_embedding = torch.load(embedding_path)

      avgbert_embedding_values = torch.stack(list(avgbert_embedding.values()))

      print('data loaded')

      bertflowdataset = BERTFLOWDATASET(avgbert_embedding_values)

      bertflowdataloader = DataLoader(bertflowdataset,batch_size = training_batch_size,
                                 shuffle = True)

      glow = Glow(image_shape = (7,12,64),
                  hidden_channels = 32,
                  K = 16,
                  L = 1,
                  actnorm_scale = 1.0,
                  flow_permutation = 'shuffle',
                  flow_coupling = 'affine',
                  LU_decomposed = False,
                  split = True).to(device)

      optimizer = torch.optim.AdamW(glow.parameters(), lr=1e-3)

      max_train_steps = num_epoches*len(bertflowdataloader)

      loss_list = []

      print('start training')

      progress_bar = tqdm(range(max_train_steps))

      glow.train()

      for i in range(num_epoches):
          
          for data_batch in bertflowdataloader:
              
              optimizer.zero_grad()
              
              embedding = data_batch.to(device)
              
              z,nll = glow(embedding.permute(0,3,1,2),None)
              
              loss = torch.mean(nll)
              
              loss_list.append(loss.item())
              
              loss.backward()
              
              optimizer.step()
              
              progress_bar.update(1)

      print('training finished')

      torch.save(glow.state_dict(), '{0}/{1}'.format(model_save_path,model_save_name))

      print('model saved')

      print('start to produce flow embeddings')

      total_num = np.ceil(avgbert_embedding_values.shape[0]/processing_batch_size).astype(int)

      bertflow_embedding_list = []

      _ = glow.eval()

      progress_bar = tqdm(range(total_num))

      for i in range(total_num):
          
          lower = i*processing_batch_size

          upper = (i+1)*processing_batch_size
          
          with torch.no_grad():
              
              embedding = avgbert_embedding_values[lower:upper,:].to(device)
              
              z,_ = glow(embedding.permute(0,3,1,2),None)
              
          bertflow_embedding_list.append(z.squeeze().detach().cpu())
          
          progress_bar.update(1)
          
      bertflow_embedding_total = torch.cat(bertflow_embedding_list,axis = 0)

      bertflow_embedding = dict(zip(list(avgbert_embedding.keys()),bertflow_embedding_total))

      print('flow embeddings produced')

      torch.save(bertflow_embedding,'{0}/{1}'.format(embedding_save_path,embedding_save_name))

      print('flow embeddings saved')
