import os
import numpy as np
from finetune import finetune_bert
from producing_embeddings import produce_embeddings
from bert_flow import bert_flow
from pos_processing import pos_processing
from glda_model import glda_model
from corex_model import corex_model
from zsl import zsl_model
from bert_flow_vae import bert_flow_vae
from tf_idf_process import tf_idf_processing
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, type=str, help='filename of the dataset (csv file)')
    parser.add_argument('-columnname', required=True, type=str, help='column name of the texts in the csv file')
    parser.add_argument('-seedwords', required=True, type=str, help='filename of the seed words')
    parser.add_argument('-labelnames', required=True, type=str, help='filename of the label surface names')
    parser.add_argument('--topic_model', type=str, help='backend model. could be "glda", "corex", "zsl" or "corex+zsl"',default='corex+zsl')
    parser.add_argument('--dataset_path', type=str, help='path to dataset',default='dataset')
    parser.add_argument('--pooling', type=str, help='pooling strategy of sentence embeddings',default='tfidf')
    parser.add_argument('--flow',help = 'whether to use flow calibration', action='store_true')
    parser.add_argument('--hps', help='whether to use hyper-parameter scheduling',action='store_true')
    parser.add_argument('--training_batch_size', type=int, help='training batch size (for models with training process)',default=512)
    parser.add_argument('--processing_batch_size', type=int, help='processing batch size (for models without training process)',default=1024)
    parser.add_argument('--random_state_topic_model', type=int, help='random seed of the topic model (for repeatability)',default=0)
    parser.add_argument('--num_epoches_bert', type=int, help='number of epoches for bert',default=10)
    parser.add_argument('--num_epoches_flow', type=int, help='number of epoches for bert-flow',default=10)
    parser.add_argument('--num_epoches_vae', type=int, help='number of epoches for bert-flow-vae',default=10)
    parser.add_argument('--gamma', type=float, help='control the ratio in the loss',default=1)
    parser.add_argument('--omega', type=float, help='control the weights of the mixture of backend models',default=0.5)
    args = parser.parse_args()
    
    dataset = args.dataset
    text_name = args.columnname
    seedwords = args.seedwords
    labelnames = args.labelnames

    topic_model = args.topic_model
    dataset_path = args.dataset_path
    flow = args.flow
    pooling = args.pooling
    hps = args.hps
    training_batch_size = args.training_batch_size
    processing_batch_size = args.processing_batch_size
    random_state_topic_model = args.random_state_topic_model
    num_epoches_bert = args.num_epoches_bert
    num_epoches_flow = args.num_epoches_flow
    num_epoches_vae = args.num_epoches_vae
    gamma = args.gamma
    omega = args.omega

    with open(seedwords,'r') as f:

        seed_words = eval(f.read())

    with open(labelnames,'r') as f:

        label_names = eval(f.read())

    BERT_DIM = 768
    HIDDEN_DIM = 512
    FC_DIM = 768
    DIMENSION_SEQUENCE = [96,196,384,512,768]

    num_total_topics = len(seed_words)

    if not os.path.exists('temp'):
        os.mkdir('temp')

    temp_path = 'temp'

    bert_finetuned_model = 'bert_finetuned.pt'
    bert_embeddings = 'bert_embeddings.pt'

    bert_flow_model = 'bert_flow.pt'
    bert_flow_embeddings = 'bert_flow_embedding.pt'

    bert_flow_vae_model = 'bert_flow_vae.pt'
    bert_flow_vae_pred = 'bert_flow_vae_pred.npy'

    masked_tokens = 'masked_tokens.npy'
    processed_documents = 'processed_documents.txt'
    tfidf_save_name = 'tfidf.pkl'

    glda_res = 'glda_res.npy'
    corex_res = 'corex_res.npy'
    zsl_res = 'zsl_res.npy'
    corex_zsl_res = 'corex+zsl_res.npy'

    if not os.path.exists('{0}/{1}'.format(temp_path,processed_documents)):

        pos_processing(
            dataset_path = '{0}/{1}'.format(dataset_path,dataset),
            text_name = text_name,
            tokens_save_path = temp_path,
            tokens_save_name = masked_tokens,
            documents_save_path = temp_path,
            documents_save_name = processed_documents,
            processing_batch_size = processing_batch_size,
            )

        print('finished pos-processing')

    else:

        print('pos done')

    if not os.path.exists('{0}/{1}'.format(temp_path,bert_finetuned_model)):

        finetune_bert(
            dataset_path = '{0}/{1}'.format(dataset_path,dataset),
            text_name = text_name,
            model_save_path = temp_path,
            model_save_name = bert_finetuned_model,
            num_epoches = num_epoches_bert
        )

        print('finished finetuning bert')

    else:

        print('finetune done')

    if not os.path.exists('{0}/{1}'.format(temp_path,tfidf_save_name)):

        tf_idf_processing(
            '{0}/{1}'.format(dataset_path,dataset),
            text_name = text_name,
            processing_batch_size = processing_batch_size,
            tfidf_save_path = temp_path,
            tfidf_save_name = tfidf_save_name
        )

        print('finished tf-idf processing')

    else:

        print('tf-idf done')

    if not os.path.exists('{0}/{1}'.format(temp_path,bert_embeddings)):

        produce_embeddings(
            dataset_path = '{0}/{1}'.format(dataset_path,dataset),
            model_path = '{0}/{1}'.format(temp_path,bert_finetuned_model),
            text_name = text_name,
            processing_batch_size = training_batch_size,
            embedding_save_path = temp_path,
            embedding_save_name = bert_embeddings,
            pooling = pooling,
            tf_idf_model_path = '{0}/{1}'.format(temp_path,tfidf_save_name)
        )

        print('finished producing bert-embeddings')

    else:

        print('embeddings produced')

    if flow :
        if not os.path.exists('{0}/{1}'.format(temp_path,bert_flow_embeddings)):
            bert_flow(
                embedding_path = '{0}/{1}'.format(temp_path,bert_embeddings),
                model_save_path = temp_path,
                model_save_name = bert_flow_model,
                embedding_save_path = temp_path,
                embedding_save_name = bert_flow_embeddings,
                training_batch_size = training_batch_size,
                num_epoches = num_epoches_flow,
                processing_batch_size = processing_batch_size,
                )
            print('finished bert-flow-embeddings')
        else:
            print('flow calibrated')
        

    if topic_model == 'glda':

        if not os.path.exists('{0}/{1}'.format(temp_path,glda_res)):

            glda_model(
                seed_words = seed_words,
                num_total_topics = num_total_topics,
                tokens_path = '{0}/{1}'.format(temp_path,masked_tokens),
                documents_path = '{0}/{1}'.format(temp_path,processed_documents),
                random_state = random_state_topic_model,
                glda_res_save_path = temp_path,
                glda_res_save_name = glda_res,
                )
            
            topic_model_res = glda_res
            print('finished producing glda results')

        else:
            topic_model_res = glda_res
            print('glda finished')
        
    elif topic_model == 'corex':

        if not os.path.exists('{0}/{1}'.format(temp_path,corex_res)):

            corex_model(
                anchor_words = seed_words,
                num_total_topics = num_total_topics,
                documents_path = '{0}/{1}'.format(temp_path,processed_documents),
                random_state = random_state_topic_model,
                corex_res_save_path = temp_path,
                corex_res_save_name = corex_res,
                )
            topic_model_res = corex_res
            print('finished producing corex results')

        else:
            topic_model_res = corex_res
            print('corex finished')
        
    elif topic_model == 'zsl':

        if not os.path.exists('{0}/{1}'.format(temp_path,zsl_res)):

            zsl_model(
                dataset_path = '{0}/{1}'.format(dataset_path,dataset),
                label_name = label_names,
                text_name = text_name,
                processing_batch_size = processing_batch_size//10, # save VRAM
                zsl_res_save_path = temp_path,
                zsl_res_save_name = zsl_res,
                cut = False,
            )
            topic_model_res = zsl_res
            print('finished producing zsl results')

        else:
            topic_model_res = zsl_res
            print('zsl finished')
        
    elif topic_model == 'corex+zsl':

        if not os.path.exists('{0}/{1}'.format(temp_path,corex_zsl_res)):
            
            if not os.path.exists('{0}/{1}'.format(temp_path,zsl_res)):

                zsl_model(
                    dataset_path = '{0}/{1}'.format(dataset_path,dataset),
                    label_name = label_names,
                    text_name = text_name,
                    processing_batch_size = processing_batch_size//10, # save VRAM
                    zsl_res_save_path = temp_path,
                    zsl_res_save_name = zsl_res,
                    cut = False,
                )
                
            if not os.path.exists('{0}/{1}'.format(temp_path,corex_res)):
                
                corex_model(
                    anchor_words = seed_words,
                    num_total_topics = num_total_topics,
                    documents_path = '{0}/{1}'.format(temp_path,processed_documents),
                    random_state = random_state_topic_model,
                    corex_res_save_path = temp_path,
                    corex_res_save_name = corex_res,
                    )
            
            corex_res = np.load('{0}/{1}'.format(temp_path,corex_res))
            
            zsl_res = np.load('{0}/{1}'.format(temp_path,zsl_res))
            
            probs = omega*zsl_res+(1-omega)*corex_res
            
            np.save('{0}/{1}'.format(temp_path,corex_zsl_res),probs)
            
            topic_model_res = corex_zsl_res

            print('corex+zsl saved')

        else:
            topic_model_res = corex_zsl_res
            print('corex+zsl finished')

    bert_flow_vae(
                embedding_path = '{0}/{1}'.format(temp_path,bert_flow_embeddings),
                topic_res_path = '{0}/{1}'.format(temp_path,topic_model_res),
                model_save_path = temp_path,
                model_save_name = bert_flow_vae_model,
                bert_flow_vae_pred_path = temp_path,
                bert_flow_vae_pred_name = bert_flow_vae_pred,
                hps = hps,
                num_epoches = num_epoches_vae,
                gamma = gamma,
                training_batch_size = training_batch_size,
                BERT_DIM = BERT_DIM,
                HIDDEN_DIM = HIDDEN_DIM,
                FC_DIM = FC_DIM,
                DIMENSION_SEQUENCE = DIMENSION_SEQUENCE,
                processing_batch_size = processing_batch_size,
                )
    print('finished bert-flow-vae')
