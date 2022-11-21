import json, os
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="Experiment directory")
    parser.add_argument("check_path", type=str, help="Test checkpoint path")
    args = parser.parse_args()
    file_path = os.path.join(args.exp_dir, 'hyper-p.json')
    hyper_p = json.load(open(file_path, 'r'))
    hyper_p['inference']['infer']['option']['resume'] = args.check_path
    json.dump(hyper_p, open(file_path, 'w'), indent=4)