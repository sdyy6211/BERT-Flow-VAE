def corex_model(
      anchor_words,
      num_total_topics,
      documents_path,
      random_state,
      corex_res_save_path,
      corex_res_save_name,
      ):
      
      import numpy as np
      from corextopic import corextopic as ct
      from sklearn.feature_extraction.text import CountVectorizer

      with open(documents_path,'r',encoding = 'utf-8') as f:
            
          documents = eval(f.read())

      print('data loaded')

      dic = set([w for d in documents for w in d.split()])

      check_words = [[True for w in d if w in dic] for d in anchor_words]
    
      assert(all(check_words))

      countvec = CountVectorizer()

      docvec = countvec.fit_transform(documents)

      anchor_idx = [[countvec.vocabulary_[j] for j in i] for i in anchor_words]

      topic_model = ct.Corex(n_hidden=num_total_topics,seed = random_state)

      print('fitting the model')

      topic_model.fit(docvec, anchors=anchor_idx, anchor_strength=4)

      print('model fitted')

      print('producing results')

      pred = np.exp(topic_model.log_p_y_given_x)

      print('results produced')

      np.save('{0}/{1}'.format(corex_res_save_path,corex_res_save_name),pred)

      print('results saved')
