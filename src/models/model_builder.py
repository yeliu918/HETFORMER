
import torch
import torch.nn as nn
from longformer import Heterformer, HeterformerConfig
from torch.nn.init import xavier_uniform_
from models.optimizers import Optimizer
from models.encoder import Classifier

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_longformer = False, longformer_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.dropout = nn.Dropout(0.1)
        # self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        load_pretrained_longformer= "../pretrained_model/longformer-base-4096"
        config = HeterformerConfig.from_pretrained(load_pretrained_longformer)
        config.attention_window = args.attention_window
        config.attention_mode = "sliding_chunks"
        self.heterformer = Heterformer.from_pretrained(load_pretrained_longformer, config=config)
        self.encoder = Classifier(config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, entity_mask, segs, clss, mask, mask_cls, sentence_range=None):
        attention_mask = torch.ones(x.shape, dtype=torch.long, device=x.device)
        attention_mask[mask == False] = 0
        attention_mask[x == 0] = 2 #tokenizer.bos_token_id
        attention_mask[x == 1] = 0  #tokenizer.pad_token_id
        encoded_layers, _ = self.heterformer(x, attention_mask= attention_mask, entity_mask= entity_mask)
        top_vec = encoded_layers
        top_vec = self.dropout(top_vec)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
