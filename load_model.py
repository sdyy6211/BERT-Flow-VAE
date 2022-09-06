def load_model(
    LATENT_DIM,
    bert_model_path,
    tf_idf_model_path,
    glow_model_path,
    bert_vae_model_path,
    use_tfidf
):    
    import torch
    from transformers import AutoModel
    import pickle
    from models import AVGBERT, Glow, Encoder, Decoder, VAEModel, BFV
    from utils import identity_token, produce_tfidf_scores, tfidf_mean_pooling, mean_pooling

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert = AutoModel.from_pretrained('distilbert-base-uncased').to(device)

    bert.load_state_dict(torch.load(bert_model_path))

    avgbert = AVGBERT(bert)

    with open(tf_idf_model_path,'rb') as f:

        tfidf = pickle.load(f)

    glow = Glow(image_shape = (7,12,64),
                hidden_channels = 32,
                K = 16,
                L = 1,
                actnorm_scale = 1.0,
                flow_permutation = 'shuffle',
                flow_coupling = 'affine',
                LU_decomposed = False,
                split = True).to(device)

    glow.load_state_dict(torch.load(glow_model_path))
    glow.set_actnorm_init()
    glow = glow.eval()

    BERT_DIM = 768
    HIDDEN_DIM = 512
    FC_DIM = 768
    DIMENSION_SEQUENCE = [96,196,384,512,768]

    encoder = Encoder(
                    BERT_DIM,
                    HIDDEN_DIM,
                    FC_DIM,
                    LATENT_DIM
                    )

    decoder = Decoder(
                    LATENT_DIM,
                    DIMENSION_SEQUENCE,
                    BERT_DIM
                    )

    vae_model = VAEModel(encoder,decoder).to(device)

    vae_model.load_state_dict(torch.load(bert_vae_model_path))

    bfv = BFV(avgbert,tfidf,glow,vae_model.Encoder,use_tfidf)

    return bfv