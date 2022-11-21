import torch
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str, help="Input checkpoint path")
    parser.add_argument("out_path", type=str, help="Ouput checkpoint path")
    parser.add_argument("stop_steps", type=int, help="New stop steps")
    args = parser.parse_args()
    check = torch.load(args.in_path, 'cpu')
    for key in check:
        if 'scheduler' in key:
            check[key]['stop_step']=args.stop_steps
            check[key]['_in_stop_step']=True
    torch.save(check, args.out_path)
