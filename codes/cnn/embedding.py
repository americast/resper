from utils import *

class embed(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size):
        super().__init__()

        # architecture
        for model, dim in EMBED.items():
            if model == "char-cnn":
                self.char_embed = self.cnn(char_vocab_size, dim)
            elif model == "char-rnn":
                self.char_embed = self.rnn(char_vocab_size, dim)
            if model == "lookup":
                self.word_embed = nn.Embedding(word_vocab_size, dim, padding_idx = PAD_IDX)
            elif model == "sae":
                self.word_embed = self.sae(word_vocab_size, dim)

        if CUDA:
            self = self.cuda()

    def forward(self, xc, xw):
        hc = self.char_embed(xc) if "char-cnn" in EMBED or "char-rnn" in EMBED else None
        hw = self.word_embed(xw) if "lookup" in EMBED or "sae" in EMBED else None
        h = torch.cat([h for h in [hc, hw] if type(h) == torch.Tensor], 2)
        return h

    class cnn(nn.Module):
        def __init__(self, vocab_size, embed_size):
            super().__init__()
            dim = 50
            num_featmaps = 50 # feature maps generated by each kernel
            kernel_sizes = [3]

            # architecture
            self.embed = nn.Embedding(vocab_size, dim, padding_idx = PAD_IDX)
            self.conv = nn.ModuleList([nn.Conv2d(
                in_channels = 1, # Ci
                out_channels = num_featmaps, # Co
                kernel_size = (i, dim) # height, width
            ) for i in kernel_sizes]) # num_kernels (K)
            self.dropout = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(len(kernel_sizes) * num_featmaps, embed_size)

        def forward(self, x):
            x = x.view(-1, x.size(2)) # [batch_size (B) * word_seq_len (Lw), char_seq_len (Lc)]
            x = self.embed(x) # [B * Lw, Lc, dim (H)]
            x = x.unsqueeze(1) # [B * Lw, Ci, Lc, W]
            h = [conv(x) for conv in self.conv] # [B * Lw, Co, Lc, 1] * K
            h = [F.relu(k).squeeze(3) for k in h] # [B * Lw, Co, Lc] * K
            h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h] # [B * Lw, Co] * K
            h = torch.cat(h, 1) # [B * Lw, Co * K]
            h = self.dropout(h)
            h = self.fc(h) # fully connected layer [B * Lw, embed_size]
            h = h.view(BATCH_SIZE, -1, h.size(1)) # [B, Lw, embed_size]
            return h

    class rnn(nn.Module):
        def __init__(self, vocab_size, embed_size):
            super().__init__()
            self.dim = embed_size
            self.rnn_type = "GRU" # LSTM, GRU
            self.num_dirs = 2 # unidirectional: 1, bidirectional: 2
            self.num_layers = 2

            # architecture
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = PAD_IDX)
            self.rnn = getattr(nn, self.rnn_type)(
                input_size = self.dim,
                hidden_size = self.dim // self.num_dirs,
                num_layers = self.num_layers,
                bias = True,
                batch_first = True,
                dropout = DROPOUT,
                bidirectional = self.num_dirs == 2
            )

        def init_state(self, b): # initialize RNN states
            n = self.num_layers * self.num_dirs
            h = self.dim // self.num_dirs
            hs = zeros(n, b, h) # hidden state
            if self.rnn_type == "LSTM":
                cs = zeros(n, b, h) # LSTM cell state
                return (hs, cs)
            return hs

        def forward(self, x):
            s = self.init_state(x.size(0) * x.size(1))
            x = x.view(-1, x.size(2)) # [batch_size (B) * word_seq_len (Lw), char_seq_len (Lc)]
            x = self.embed(x) # [B * Lw, Lc, embed_size (H)]
            h, s = self.rnn(x, s)
            h = s if self.rnn_type == "GRU" else s[-1]
            h = torch.cat([x for x in h[-self.num_dirs:]], 1) # final hidden state [B * Lw, H]
            h = h.view(BATCH_SIZE, -1, h.size(1)) # [B, Lw, H]
            return h

    class sae(nn.Module): # self attentive encoder
        def __init__(self, vocab_size, embed_size = 512):
            super().__init__()
            dim = embed_size
            num_layers = 1

            # architecture
            self.embed = nn.Embedding(vocab_size, dim, padding_idx = PAD_IDX)
            self.pe = self.pos_encoding(dim)
            self.layers = nn.ModuleList([self.layer(dim) for _ in range(num_layers)])

        def forward(self, x):
            mask = self.maskset(x)
            x = self.embed(x)
            h = x + self.pe[:x.size(1)]
            for layer in self.layers:
                h = layer(h, mask[0])
            return h

        @staticmethod
        def maskset(x): # set of mask and lengths
            mask = x.eq(PAD_IDX)
            return (mask.view(BATCH_SIZE, 1, 1, -1), x.size(1) - mask.sum(1))

        @staticmethod
        def pos_encoding(dim, maxlen = 1000): # positional encoding
            pe = Tensor(maxlen, dim)
            pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
            k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2.) / dim)
            pe[:, 0::2] = torch.sin(pos * k)
            pe[:, 1::2] = torch.cos(pos * k)
            return pe

        class layer(nn.Module): # encoder layer
            def __init__(self, dim):
                super().__init__()

                # architecture
                self.attn = embed.sae.attn_mh(dim)
                self.ffn = embed.sae.ffn(dim)

            def forward(self, x, mask):
                z = self.attn(x, x, x, mask)
                z = self.ffn(z)
                return z

        class attn_mh(nn.Module): # multi-head attention
            def __init__(self, dim):
                super().__init__()
                self.D = dim # dimension of model
                self.H = 8 # number of heads
                self.Dk = self.D // self.H # dimension of key
                self.Dv = self.D // self.H # dimension of value

                # architecture
                self.Wq = nn.Linear(self.D, self.H * self.Dk) # query
                self.Wk = nn.Linear(self.D, self.H * self.Dk) # key for attention distribution
                self.Wv = nn.Linear(self.D, self.H * self.Dv) # value for context representation
                self.Wo = nn.Linear(self.H * self.Dv, self.D)
                self.dropout = nn.Dropout(DROPOUT)
                self.norm = nn.LayerNorm(self.D)

            def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
                c = np.sqrt(self.Dk) # scale factor
                a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
                a = a.masked_fill(mask, -10000) # masking in log space
                a = F.softmax(a, -1)
                a = torch.matmul(a, v)
                return a # attention weights

            def forward(self, q, k, v, mask):
                x = q # identity
                q = self.Wq(q).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
                k = self.Wk(k).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
                v = self.Wv(v).view(BATCH_SIZE, -1, self.H, self.Dv).transpose(1, 2)
                z = self.attn_sdp(q, k, v, mask)
                z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, self.H * self.Dv)
                z = self.Wo(z)
                z = self.norm(x + self.dropout(z)) # residual connection and dropout
                return z

        class ffn(nn.Module): # position-wise feed-forward networks
            def __init__(self, dim):
                super().__init__()
                dim_ffn = 2048

                # architecture
                self.layers = nn.Sequential(
                    nn.Linear(dim, dim_ffn),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(dim_ffn, dim)
                )
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                z = x + self.layers(x) # residual connection
                z = self.norm(z) # layer normalization
                return z