# config_adapter.py
from utils.unified_config import get_unified_args

_unified_args = None

def get_config():
    global _unified_args
    if _unified_args is None:
        _unified_args = get_unified_args()
    
    class Config:
        def __init__(self, unified_args):
            self.backbone = unified_args.backbone
            self.backbone_out_features = unified_args.backbone_out_features
            self.normalize_img_features = unified_args.normalize_img_features
            self.lr = unified_args.lr
            self.weight_decay = unified_args.weight_decay
            self.n_epochs = unified_args.n_epochs
            self.batch_size = unified_args.batch_size
            self.eval_frequency = unified_args.eval_frequency
            self.data_dir = unified_args.data_dir
            self.dataset = unified_args.dataset
            self.n_workers = unified_args.n_workers
            self.n_bits = unified_args.n_bits
            self.topk = unified_args.topk
            self.method = unified_args.method
            self.distance_loss = unified_args.distance_loss
            self.margin_beta = unified_args.margin_beta
            self.margin_m_loss = unified_args.margin_m_loss
            self.type_of_triplets = unified_args.type_of_triplets
            self.automargin_mode = unified_args.automargin_mode
            self.k_param_automargin = unified_args.k_param_automargin
            self.k_n_param_autobeta = unified_args.k_n_param_autobeta
            self.k_p_param_autobeta = unified_args.k_p_param_autobeta
            self.loss_w_lambda = unified_args.loss_w_lambda
            self.loss_w_neg = unified_args.loss_w_neg
    
    return Config(_unified_args)