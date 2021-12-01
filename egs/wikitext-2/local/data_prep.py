import os
try:
    import torchtext
except ModuleNotFoundError as me:
    print(me)
    print("Install torchtext with command:")
    print("pip install -U torchtext")
    exit(1)

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer


os.makedirs('data', exist_ok=True)
'''
Use basic english tokenizer from pytorch,
for more details of what tokenizer does,
see _basic_english_normalize(), at
https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
'''
tokenizer = get_tokenizer('basic_english')

data_iter = WikiText2()
dataset = ['train', 'test', 'valid']

for d_iter, d_set in zip(data_iter, dataset):
    f_text = os.path.join('data', d_set)
    with open(f_text, 'w') as fo:
        for item in d_iter:
            fo.write(' '.join(tokenizer(item))+'\n')
    print(f"> Normalized text saved at {f_text}")
