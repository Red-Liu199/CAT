
import os
import kaldiio
from kaldiio import ReadHelper, WriteHelper

if __name__ == "__main__":

    src_path = '/mnt/workspace'
    dst_path = '/home'
    for file in ['tr', 'cv', 'eval92', 'dev93']:
        scppath = 'data/all_ark/{}.scp'.format(file)

        assert os.path.isfile(scppath), f"{scppath}"

        with open(scppath, 'r') as fi:
            with open('./'+scppath.split('/')[-1], 'w') as fo:
                for line in fi:
                    key, loc_ark = line.split()

                    loc_ark = loc_ark.replace(src_path, dst_path)
                    fo.write('{} {}\n'.format(key, loc_ark))
