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

# change dir to a different one to test whether cat module has been installed.
$(cd egs && python -c "import cat" >/dev/null 2>&1) || (
    python -m pip install -r requirements.txt || exit 1

    # check installation
    $(cd egs && python -c "import cat") >/dev/null
)
echo "install module:cat and its requirements done."

# install ctcdecode is annoying...
$(python -c "import ctcdecode" >/dev/null 2>&1) || (
    [ ! -d src/ctcdecode ] && (
        git clone --recursive git@github.com:maxwellzh/ctcdecode.git src/ctcdecode
    )

    # ctcdecode doesn't support -e, but we want to install locally
    # ... so we cannot put it in requirements.txt
    python -m pip install src/ctcdecode || exit 1
)
echo "install module:ctcdecode done."

# install kenlm
! [[ -x "$(command -v src/bin/lmplz)" && -x "$(command -v src/bin/build_binary)" ]] && (
    [ ! -d src/kenlm ] && (
        echo "No 'src/kenlm' folder found, seems you haven't done previous installation."
        echo "... Or you've installed the module somewhere else, then please copy the src/bin to current repo/src/bin"
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
)
echo "install module:kenlm done."

echo "$0 done"
exit 0
