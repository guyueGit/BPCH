# args_adapter.py
from utils.unified_config import get_unified_args
import math
import pandas as pd

_unified_args = None

def get_args():
    global _unified_args
    if _unified_args is None:
        _unified_args = get_unified_args()
    
    class Args:
        def __init__(self, unified_args):
            self.save_dir = unified_args.save_dir
            self.clip_path = unified_args.clip_path
            self.pretrained = unified_args.pretrained
            self.dataset = unified_args.dataset
            self.index_file = unified_args.index_file
            self.caption_file = unified_args.caption_file
            self.label_file = unified_args.label_file
            self.output_dim = unified_args.output_dim
            self.numclass = unified_args.numclass
            self.epochs = unified_args.epochs
            self.max_words = unified_args.max_words
            self.resolution = unified_args.resolution
            self.batch_size = unified_args.batch_size
            self.num_workers = unified_args.num_workers
            self.query_num = unified_args.query_num
            self.train_num = unified_args.train_num
            self.lr_decay_freq = unified_args.lr_decay_freq
            self.display_step = unified_args.display_step
            self.seed = unified_args.seed
            self.hypseed = unified_args.hypseed
            self.lr = unified_args.lr
            self.alpha = unified_args.alpha
            self.lr_decay = unified_args.lr_decay
            self.clip_lr = unified_args.clip_lr
            self.weight_decay = unified_args.weight_decay
            self.warmup_proportion = unified_args.warmup_proportion
            self.is_train = unified_args.is_train.lower() == "true"

            
            self.threshold = self.calculate_threshold()
        
        def calculate_threshold(self):
            try:
                excel_path = '/remote-home/zhangli/hurui/BPCH/utils/codetable.xlsx'
                df = pd.read_excel(excel_path, header=None, engine='openpyxl')
                
                col_index = math.ceil(math.log(self.numclass, 2))
                
                return df.iloc[self.output_dim, col_index]
            except Exception as e:
                print(f"Error calculating threshold: {str(e)}")
                return 0.0 
    
    return Args(_unified_args)