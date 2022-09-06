def glda_model(
      seed_words,
      num_total_topics,
      tokens_path,
      documents_path,
      random_state,
      glda_res_save_path,
      glda_res_save_name,
      ):
      
      import numpy as np
      from models import GLDA

      maskedtokens = np.load(tokens_path)

      with open(documents_path,'r',encoding = 'utf-8') as f:
            
          documents = eval(f.read())

      print('data loaded')

      dic = set([w for d in documents for w in d.split()])

      check_words = [[True for w in d if w in dic] for d in seed_words]
    
      assert(all(check_words))

      glda = GLDA(documents,num_total_topics,seed_words,len(seed_words)/num_total_topics,random_state = random_state)

      print('fitting the model')

      glda.fit()

      print('model fitted')

      print('producing results')

      glda_res = glda.transform(documents)

      normalizer = (maskedtokens!=0).sum(axis = 1).reshape(-1,1)+1e-4

      normalizer = np.repeat(normalizer,num_total_topics).reshape(-1,num_total_topics)

      probs = (glda_res-1/num_total_topics)/normalizer

      probs = np.clip(0,1,probs)

      print('results produced')

      np.save('{0}/{1}'.format(glda_res_save_path,glda_res_save_name),probs)

      print('results saved')
