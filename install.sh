#!/bin/bash
#
# install the requirement packages
set -e -u
<<"PARSER"
('-p', "--package", type=str, choices=['all', 'cat', 'ctcdecode', 'kenlm'], default=['all'], nargs='*',
    help="Select modules to be installed/uninstalled. Default: all.")
('-r', "--remove", action='store_true', default=False,
    help="Remove modules instead of installing.")
PARSER
eval $(python cat/utils/parseopt.py $0 $*)

function exc_install() {
    name=$1
    [ -z $name ] && {
        echo "error calling exc_install()"
        return 1
    }

    case $name in
    cat | all)
        # change dir to a different one to test whether cat module has been installed.
        $(cd egs && python -c "import cat" >/dev/null 2>&1) || {
            python -m pip install -r requirements.txt || return 1

            # check installation
            $(cd egs && python -c "import cat") >/dev/null || return 1
        }
        ;;&
    ctcdecode | all)
        # install ctcdecode is annoying...
        $(python -c "import ctcdecode" >/dev/null 2>&1) || {
            [ ! -d src/ctcdecode ] &&
                git clone --recursive git@github.com:maxwellzh/ctcdecode.git src/ctcdecode

            # ctcdecode doesn't support -e, but we want to install locally
            # ... so we cannot put it in requirements.txt
            python -m pip install src/ctcdecode || return 1
        }
        ;;&
    kenlm | all)
        # install kenlm
        # kenlm is a denpendency of cat, so we first check the python installation
        $(python -c "import kenlm" >/dev/null 2>&1) ||
            python -m pip install -e git+ssh://git@github.com/kpu/kenlm.git#egg=kenlm

        ! [[ -x "$(command -v src/bin/lmplz)" && -x "$(command -v src/bin/build_binary)" ]] && {
            [ ! -d src/kenlm ] && {
                echo "No 'src/kenlm' folder found, seems you haven't done previous installation."
                echo "... Or you've installed the module somewhere else, then please copy the src/bin to current repo/src/bin"
                return 1
            }

            cd src/kenlm
            mkdir -p build && cd build
            (cmake .. && make -j $(nproc)) || {
                echo "If you meet building error and have no idea why it is raised, "
                echo "... please first confirm all requirements are installed. See"
                echo "... https://github.com/kpu/kenlm/blob/master/BUILDING"
                return 1
            }

            # link executable binary files
            cd ../../
            mkdir -p bin && cd bin
            ln -snf ../kenlm/build/bin/* ./ && cd ../../
        }
        ;;
    *) ;;
    esac

    echo "installed module:$name"
    return 0
}

function exc_rm() {
    name=$1
    [ -z $name ] && return 0

    case $name in
    cat | all)
        python -m pip uninstall -y cat
        python setup.py clean --all
        # clean building dependencies
        pip uninstall -y gather warp_rnnt webdataset kenlm
        rm -rf src/{gather,warp-rnnt,webdataset,kenlm}
        ;;&
    ctcdecode | all)
        python -m pip uninstall -y ctcdecode
        rm -rf src/ctcdecode
        ;;&
    kenlm | all)
        python -m pip uninstall -y kenlm

        [ -d src ] && {
            cd src/
            [ -d kenlm/build/bin ] && {
                for exc in $(ls kenlm/build/bin); do
                    rm -if bin/$exc
                done
            }
            [ -d kenlm ] && rm -rf kenlm/
            cd - >/dev/null
        }
        ;;
    *) ;;
    esac

    echo "removed module:$name"
    return 0
}

# squeeze the packages to 'all' once there is 'all'
for p in $package; do
    if [ $p == "all" ]; then
        package="all"
        break
    fi
done

# environment check
## python
[ ! $(command -v python) ] && {
    echo "No python interpreter in your PATH"
    exit 1
}

## python>3
[ $(python --version | awk '{print $2}' | cut -d '.' -f 1) -ne 3 ] && {
    echo "Require python3+, instead $(python --version)"
    exit 1
}

log_dir="logs"
mkdir -p $log_dir
if [ $remove == "False" ]; then
    # install packages
    for p in $package; do
        f_log="$log_dir/$p.log"
        touch $f_log
        echo "logging at $f_log"
        exc_install $p >$f_log 2>&1 || {
            echo "failed to install $p, check the log at $f_log"
            echo "clean installation..."
            $0 $* -r >/dev/null
            exit 1
        }
    done
elif [ $remove == "True" ]; then
    # remove packages
    for p in $package; do
        f_log="$log_dir/remove.$p.log"
        touch $f_log
        echo "logging at $f_log"
        exc_rm $p >$f_log 2>&1 || {
            echo "failed to remove $p, check the log at $f_log"
            exit 1
        }
    done
fi

echo "$0 done"
exit 0
