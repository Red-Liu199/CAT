import subprocess
import argparse
import re
import os

def find_best_alpha_beta(args):
    process = subprocess.run(
        "python utils/lm/lmweight_search.py "
        f"{args.dev_asr_nbest} {args.dev_lm_nbest} {args.dev_beta_nbest} "
        f"--search 0 1 1 --ground {args.dev_text} --cer", 
        shell=True, check=True, stdout=subprocess.PIPE)

    result = process.stdout.decode('utf-8').split('\n')[-2]
    print('Best result on dev:', result)
    pattern = re.compile(r'\(([0-9.-]*,\s[0-9.-]*)\)')
    alpha, beta = re.findall(pattern, result)[0].split(',')
    return float(alpha), float(beta)

def get_test_cer(args, alpha, beta):
    for test_asr_nbest, test_beta_nbest, test_lm_nbest, test_text in zip(args.test_asr_nbest, args.test_beta_nbest, args.test_lm_nbest, args.test_text):
        output_path = os.path.join(os.path.dirname(test_lm_nbest), f'onebest_a{alpha}_b{beta}')
        subprocess.run(
            "python utils/lm/interpolate_nbests.py "
            f"{output_path} --nbestlist {test_asr_nbest} {test_lm_nbest} {test_beta_nbest} " 
            f"--weights 1 {alpha} {beta} --one-best",
            shell=True, check=True, stdout=subprocess.PIPE)
        process = subprocess.run(
            "python utils/wer.py "
            f"{test_text} {output_path} --cer",
            shell=True, check=True, stdout=subprocess.PIPE
        )
        result = process.stdout.decode('utf-8').split('\n')[-2]
        print(f'Result on {test_asr_nbest}:', result)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_asr_nbest", type=str, default='data/aishell/outd-dev.nbest', help="ASR N-best list path on dev set")
    parser.add_argument("--dev_lm_nbest", type=str, help="LM N-best list path on dev set")
    parser.add_argument("--dev_beta_nbest", type=str, default='data/aishell/outd-dev-beta.nbest', help="Lengths list path on dev set")
    parser.add_argument("--dev_text", type=str, default='data/aishell/dev-text', help="The ground truth of dev set")

    parser.add_argument("--test_asr_nbest", nargs='+', type=str, default=['data/aishell/outd.nbest'],
                        help="ASR N-best lists path on test set")
    parser.add_argument("--test_lm_nbest", nargs='+', type=str, help="LM N-best lists path on test set")
    parser.add_argument("--test_beta_nbest", nargs='+', type=str, default=['data/aishell/beta_old.nbest'],
                        help="Lengths list path on test set")
    parser.add_argument("--test_text", nargs='+', type=str, default=['data/aishell/text'], 
                        help="The ground truth of test set")
    args = parser.parse_args()
    alpha, beta = find_best_alpha_beta(args)
    get_test_cer(args, alpha, beta)