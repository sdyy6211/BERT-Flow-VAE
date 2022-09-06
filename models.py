import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GLDA():

    from sklearn.decomposition import LatentDirichletAllocation

    class PTWGuidedLatentDirichletAllocation(LatentDirichletAllocation):

            def __init__(self, n_components=10, doc_topic_prior=None, 
                    topic_word_prior=None, learning_method="batch",
                    learning_decay=0.7, learning_offset=10.0,
                    max_iter=10, batch_size=128, 
                    evaluate_every=-1, total_samples=1000000.0,
                    perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100,
                    n_jobs=None, verbose=0, random_state=None, ptws=None):
                super().__init__(n_components, 
                                doc_topic_prior, 
                                topic_word_prior, 
                                learning_method, 
                                learning_decay, 
                                learning_offset, 
                                max_iter, 
                                batch_size, 
                                evaluate_every, 
                                total_samples, 
                                perp_tol, 
                                mean_change_tol, 
                                max_doc_update_iter, 
                                n_jobs, verbose, 
                                random_state)
                self.ptws = ptws

            def _init_latent_vars(self, n_features):
                from sklearn.utils import check_random_state
                from sklearn.decomposition._online_lda_fast import _dirichlet_expectation_2d
                """Initialize latent variables."""

                self.random_state_ = check_random_state(self.random_state)
                self.n_batch_iter_ = 1
                self.n_iter_ = 0

                if self.doc_topic_prior is None:
                    self.doc_topic_prior_ = 1. / self.n_components
                else:
                    self.doc_topic_prior_ = self.doc_topic_prior

                if self.topic_word_prior is None:
                    self.topic_word_prior_ = 1. / self.n_components
                else:
                    self.topic_word_prior_ = self.topic_word_prior

                init_gamma = 100.
                init_var = 1. / init_gamma
                # In the literature, this is called `lambda`
                self.components_ = self.random_state_.gamma(
                    init_gamma, init_var, (self.n_components, n_features))

                # Transform topic values in matrix for prior topic words
                if self.ptws is not None:
                    self.components_ = self.ptws.astype(float)

                self.exp_dirichlet_component_ = np.exp(
                    _dirichlet_expectation_2d(self.components_))

    def __init__(self,documents,total_topics_num,seed_words,seed_confidence, random_state):
                
        self.total_topics_num = total_topics_num
        self.seed_words = seed_words
        self.seed_confidence = seed_confidence
        self.countvec = CountVectorizer()
        
        self.countvec = self.countvec.fit(documents)
        self.doc_word_matrix = self.countvec.transform(documents)
        seed_topics = self.build_seed_dict(self.seed_words,self.countvec.vocabulary_)

        nzw_,ndz_,nz_ = self.guidedlda_initialize(self.doc_word_matrix,
                                        seed_topics,
                                        n_topics = self.total_topics_num,
                                        seed_confidence = self.seed_confidence,
                                        random_state = random_state)

        ptws = (nzw_+1/(self.total_topics_num)+np.random.randn(*nzw_.shape)/(self.total_topics_num**2))

        self.glda = self.PTWGuidedLatentDirichletAllocation(n_components = self.total_topics_num,
                                                            ptws = ptws,
                                                            random_state = random_state)
        
    def matrix_to_lists(self,doc_word):
        """Convert a (sparse) matrix of counts into arrays of word and doc indices
        Parameters
        ----------
        doc_word : array or sparse matrix (D, V)
            document-term matrix of counts
        Returns
        -------
        (WS, DS) : tuple of two arrays
            WS[k] contains the kth word in the corpus
            DS[k] contains the document index for the kth word
        """
        sparse = True
        try:
            # if doc_word is a scipy sparse matrix
            doc_word = doc_word.copy().tolil()
        except AttributeError:
            sparse = False

        if sparse and not np.issubdtype(doc_word.dtype, np.signedinteger):
            raise ValueError("expected sparse matrix with integer values, found float values")

        ii, jj = np.nonzero(doc_word)
        if sparse:
            ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
        else:
            ss = doc_word[ii, jj]

        n_tokens = int(doc_word.sum())
        DS = np.repeat(ii, ss).astype(np.intc)
        WS = np.empty(n_tokens, dtype=np.intc)
        startidx = 0
        for i, cnt in enumerate(ss):
            cnt = int(cnt)
            WS[startidx:startidx + cnt] = jj[i]
            startidx += cnt
        return WS, DS

    def guidedlda_initialize(self,X, seed_topics,n_topics,seed_confidence, random_state):
            """Initialize the document topic distribution.
            topic word distribution, etc.
            Parameters
            ----------
            seed_topics: type=dict, value={2:0, 256:0, 412:1, 113:1}
            """
            D, W = X.shape
            N = int(X.sum())

            beta = 0.1
            nzw_ = np.zeros((n_topics, W), dtype=np.intc) # + self.beta
            ndz_ = np.zeros((D, n_topics), dtype=np.intc) # + self.alpha
            nz_ = np.zeros(n_topics, dtype=np.intc)# + W * self.beta

            WS, DS = self.matrix_to_lists(X)
            ZS = np.empty_like(WS, dtype=np.intc)
            np.testing.assert_equal(N, len(WS))

            # seeded Initialization
            np.random.seed(random_state)
            count_testing = 0
            for i in range(N):
                w, d = WS[i], DS[i]
                if w not in seed_topics:
                    continue
                # check if seeded initialization
                if w in seed_topics and np.random.rand() < seed_confidence:
                    z_new = seed_topics[w]
                else:
                    z_new = i % n_topics
                ZS[i] = z_new
                ndz_[d, z_new] += 1
                nzw_[z_new, w] += 1
                nz_[z_new] += 1

            # Non seeded Initialization
            for i in range(N):
                w, d = WS[i], DS[i]
                if w in seed_topics:
                    continue
                z_new = i % n_topics
                ZS[i] = z_new
                ndz_[d, z_new] += 1
                nzw_[z_new, w] += 1
                nz_[z_new] += 1

            nzw_ = nzw_.astype(np.intc)
            ndz_ = ndz_.astype(np.intc)
            nz_ = nz_.astype(np.intc)

            return nzw_,ndz_,nz_
        
    def build_seed_dict(self,words,voc):
        seed_dict = dict()
        for t,w in enumerate(words):
            for word in w:
                seed_dict[voc[word]] = t
        return seed_dict

    def fit(self):

        self.glda.fit(self.doc_word_matrix)

    def transform(self,documents):
        doc_word_matrix = self.countvec.transform(documents)
        return self.glda._unnormalized_transform(doc_word_matrix)

    def analyze_topics(self,topic,top_k):
        word_count_dict = dict()
        reverse_vocabulary = {j:i for i,j in self.countvec.vocabulary_.items()}
        for each in self.glda.components_[topic,:].argsort()[::-1][:top_k]:
            word_count_dict[reverse_vocabulary[each]] = self.glda.components_[topic,each]
        return word_count_dict

    def save(self,path,name):
        
        import pickle

        with open('{0}/{1}'.format(path,name), 'wb') as f:

            f.write(pickle.dumps(self))

def compute_same_pad(kernel_size, stride):
          if isinstance(kernel_size, int):
              kernel_size = [kernel_size]

          if isinstance(stride, int):
              stride = [stride]

          assert len(stride) == len(
              kernel_size
          ), "Pass kernel size and stride both as int, or both as equal length iterable"

          return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]



def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z

class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer('indices',torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long))
        self.register_buffer('indices_inverse',torch.zeros((self.num_channels), dtype=torch.long))

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels, split = False):
        super().__init__()
        self.split = split
        if split:
            self.conv = Conv2dZeros(num_channels // 2, num_channels)
        else:
            self.conv = Conv2dZeros(num_channels, num_channels*2)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "split")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            if temperature == None:
                temperature = 1.0
            if self.split:
                z1 = input
                mean, logs = self.split2d_prior(z1)
                z2 = gaussian_sample(mean, logs, temperature)
                z = torch.cat((z1, z2), dim=1)
            else:
                mean, logs = self.split2d_prior(self.zeros)
                z = gaussian_sample(mean, logs, temperature)
            return z, logdet
        else:
            if self.split:
                z1, z2 = split_feature(input, "split")
                mean, logs = self.split2d_prior(z1)
                logdet = gaussian_likelihood(mean, logs, z2) + logdet
                return z1, logdet
            else:
                self.zeros = torch.zeros_like(input)
                mean, logs = self.split2d_prior(self.zeros)
                logdet = gaussian_likelihood(mean, logs, input) + logdet
                z = input
                return z, logdet

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        split
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C, split = split))
                if split:
                    self.output_shapes.append([-1, C//2, H, W])
                    C = C // 2
                else:
                    self.output_shapes.append([-1, C, H, W])

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        
        for layer, shape in zip(self.layers, self.output_shapes):
            
            z, logdet = layer(z, logdet, reverse=False)

        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        split
    ):
        super().__init__()
        self.flow = FlowNet(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            split = split
        )

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

    def prior(self, data):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        return split_feature(h, "split")

    def forward(self, x=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, temperature)
        else:
            return self.normal_flow(x)

    def normal_flow(self, x):
        b, c, h, w = x.shape

        z, objective = self.flow(x, reverse=False)

        mean, logs = self.prior(x)
        objective += gaussian_likelihood(mean, logs, z)

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

def attention_operation(query, key, value, mask, layer):
    
    n_heads = 12
    
    dim = 768

    bs, q_length, dim = query.size()
    k_length = key.size(1)

    dim_per_head = dim // n_heads

    mask_reshp = (bs, 1, 1, k_length)

    def shape(x):
        """separate heads"""
        return x.view(bs, -1, n_heads, dim_per_head).transpose(1, 2)

    def unshape(x):
        """group heads"""
        return x.transpose(1, 2).contiguous().view(bs, -1, n_heads * dim_per_head)

    q = shape(layer.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    k = shape(layer.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    v = shape(layer.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

    weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
    weights = layer.dropout(weights)  # (bs, n_heads, q_length, k_length)


    context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context_out = unshape(context)  # (bs, q_length, dim)
    context = layer.out_lin(context_out)  # (bs, q_length, dim)

    return (context,context_out)

class AVGBERT(torch.nn.Module):
    
    def __init__(self,bert):
        
        super().__init__()
        
        self.bert = bert
        
    def forward(self,input_ids,attention_mask):
        
        keep_hidden = []
        
        num_layer = len(list(self.bert.children())[1].layer)

        x = list(self.bert.children())[0](input_ids)
        
        keep_hidden.append(x)
        
        for index,layer in enumerate(list(self.bert.children())[1].layer):
            
            sa_output, extract = attention_operation(x,x,x,attention_mask,layer.attention)
            
            sa_output = layer.sa_layer_norm(sa_output + x)
            
            ffn_output  = layer.ffn(sa_output)
            
            ffn_output  = layer.output_layer_norm(ffn_output + sa_output)
            
            x = ffn_output
            
            keep_hidden.append(x)
                
        return keep_hidden


class Encoder(nn.Module):
    
    def __init__(self,
                bert_dim,
                hidden_dim,
                fc_dim,
                latent_dim):
        
        super().__init__()
        
        self.activation = nn.PReLU()
        
        self.linear = nn.Linear(bert_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.out = nn.Linear(hidden_dim, fc_dim)
        
        self.mean_ = nn.Linear(fc_dim,latent_dim)
        
        self.var_ = nn.Linear(fc_dim,latent_dim)
        
    def forward(self, embedded, random_):
        
        if random_:
            
            self.norm.train()
            
        else:
            
            self.norm.eval()
        
        hidden = embedded
        
        hidden = self.linear(hidden)
        
        hidden = self.norm(hidden)
            
        hidden = self.activation(hidden)
        
        output = self.out(hidden)
        
        mean_ = self.mean_(output)
        
        log_var_ = self.var_(output)
        
        return mean_,log_var_
    
class Decoder(nn.Module):
    
    def __init__(self,latent_dim,dimension_sequence,bert_dim):
        
        super().__init__()
        
        self.activation = nn.PReLU()
        
        self.linear1 = nn.Linear(latent_dim, dimension_sequence[0])
        
        self.norm = nn.LayerNorm(dimension_sequence[0])
        
        self.linear2 = nn.Linear(dimension_sequence[0], dimension_sequence[1])
        
        self.linear3 = nn.Linear(dimension_sequence[1], dimension_sequence[2])
        
        self.linear4 = nn.Linear(dimension_sequence[2], dimension_sequence[3])
        
        self.out = nn.Linear(dimension_sequence[3],bert_dim)
        
    def forward(self, x ,random_ = True):
        
        if random_:
            
            self.norm.train()
            
        else:
            
            self.norm.eval()
        
        x = self.linear1(x)
        
        x = self.norm(x)
        
        x = self.activation(x)
        
        x = self.activation(self.linear2(x))
        
        x = self.activation(self.linear3(x))
        
        x = self.activation(self.linear4(x))
        
        output = self.out(x)
        
        return output
    
class VAEModel(nn.Module):
    
    def __init__(self,Encoder,Decoder,random_ = True):
        
        super().__init__()
        
        self.Encoder = Encoder
        
        self.Decoder = Decoder
        
        self.random_ = random_
        
    def reparameterization(self, mean_, log_var_, random_):
        
        if random_:
        
            epsilon = torch.randn_like(log_var_).to(device) 

            z = mean_ + log_var_*epsilon              
            
        else:
            
            z = mean_
        
        return z
    
    def forward(self, x):
        
        mean, log_var = self.Encoder(x,random_ = self.random_)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var),random_ = self.random_) 
        
        x_hat = self.Decoder(z,random_ = self.random_)
        
        return x_hat, mean, log_var


###################################################################################################

from utils import produce_tfidf_scores, tfidf_mean_pooling, mean_pooling

class BFV(nn.Module):

    def __init__(self,avgbert,tfidf,flow,encoder,use_tfidf):

        super().__init__()

        self.avgbert = avgbert

        self.tfidf = tfidf

        self.flow = flow

        self.encoder = encoder

        self.use_tfidf = use_tfidf

    def forward(self,input_ids,attention_mask):

        embedding = torch.stack(self.avgbert(input_ids,attention_mask),axis = 2)

        if self.use_tfidf:

            weighed_attention_mask = produce_tfidf_scores(self.tfidf,input_ids,attention_mask).to(device)

            embedding = tfidf_mean_pooling(embedding,weighed_attention_mask).view(-1,7,12,64)

        else:
            
            embedding = mean_pooling(embedding,attention_mask).view(-1,7,12,64)

        embedding,_ = self.flow(embedding.permute(0,3,1,2),None)

        embedding = embedding[:,:,[0,1,5],:].mean(axis = 2).view(-1,64*12)

        mean, log_var = self.encoder(embedding,random_ = False)

        return mean