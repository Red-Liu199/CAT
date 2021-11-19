# download data at http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5
set -u

data=tmp/THUCNews

for file in $(find $data -name *.txt); do
    cat $file
done >>data/thucnews.txt
