#!/bin/bash
#
# install the requirement packages
set -e

[ ! $(command -v python) ] && (
    echo "No python interpreter in your PATH" &&
        exit 1
)

[ $(python --version | awk '{print $2}' | cut -d '.' -f 1) -ne 3 ] && (
    echo "Require python3+, instead $(python --version)" &&
        exit 1
)

# python -m pip install -r requirements.txt || exit 1

# ctcdecode requires to download many archives
# so I upload the package to Tsinghua Cloud to accelarate downloading
wget --no-verbose https://cloud.tsinghua.edu.cn/f/c6503c27828d43daafed/?dl=1 -O transducer_install_ctcdecode.tar.gz
python -m pip install transducer_install_ctcdecode.tar.gz && rm transducer_install_ctcdecode.tar.gz
echo "install module:ctcdecode done."

# install kenlm
[ ! -d src/kenlm ] && (
    echo "No 'src/kenlm' folder found, seems you haven't done previous installation."
    exit 1
)
cd src/kenlm
mkdir -p build && cd build
(cmake .. && make -j $(nproc)) || (
    echo "If you meet building error and have no idea why it is raised, "
    echo "... please first confirm all requirements are installed. See"
    echo "... https://github.com/kpu/kenlm/blob/master/BUILDING"
    exit 1
)

# link executable binary files
cd ../../
mkdir -p bin && cd bin
ln -snf ../kenlm/build/bin/* ./ && cd ../../
echo "install module:kenlm done."

echo "$0 done"
exit 0
