# unified_config.py
import argparse
def get_unified_args():
    """统一所有模块的参数解析器"""
    parser = argparse.ArgumentParser(description="Unified Configuration")
    
    parser.add_argument("--save-dir", type=str, default="./result")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--dataset", type=str, default="nuswide")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.txt")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--numclass", type=int, default=21)   
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--max-words", type=int, default=32)   
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    #parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)
    #parser.add_argument("--query-num", type=int, default=140)
    #parser.add_argument("--train-num", type=int, default=800)

    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)
    parser.add_argument("--hypseed", type=int, default=0)
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    
    parser.add_argument("--is-train", 
                        type=str,
                        choices=["True", "False"],
                        default="True",
                        help="Set training mode (True/False)")
    
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--backbone-out-features", type=int, default=512)
    parser.add_argument("--normalize-img-features", type=bool, default=True)
    
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--eval-frequency", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="../_datasets")
    parser.add_argument("--n-workers", type=int, default=4)
    
    parser.add_argument("--n-bits", type=int, default=16)
    parser.add_argument("--topk", type=int, default=1000)
    
    parser.add_argument("--method", type=str, default="AdaTriplet-AM")
    parser.add_argument("--distance-loss", type=str, default="cosine")
    
    parser.add_argument("--margin-beta", type=float, default=0)
    parser.add_argument("--margin-m-loss", type=float, default=0.25)
    
    parser.add_argument("--type-of-triplets", type=str, default="semihard")
    parser.add_argument("--automargin-mode", type=str, default="normal")
    parser.add_argument("--k-param-automargin", type=float, default=2)
    parser.add_argument("--k-n-param-autobeta", type=float, default=2)
    parser.add_argument("--k-p-param-autobeta", type=float, default=2)
    
    parser.add_argument("--loss-w-lambda", type=float, default=1)
    parser.add_argument("--loss-w-neg", type=float, default=1)
    
    return parser.parse_args()
