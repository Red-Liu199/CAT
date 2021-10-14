opts=$(python utils/arseopt.py '{
        "run":{
            "type": "str",
            "help": "Experiment names, split by ':'. Such as: \"rnnt-v1:rnnt-v2\"."
        },
        "--script":{
            "type":"str",
            "default": "run.sh",
            "help": "Name of the current running script. Default: run.sh"
        },
        "--sleep":{
            "type":"int",
            "default": 120,
            "help": "Sleep seconds between heart beats. Default: 120"
        }
    }' $0 $*) && eval $opts || exit 1

pidcur=$(ps aux | grep $script | grep -v grep | grep zhenghh | awk '{print $2}')

i=0
while [[ $(ps aux | grep $pidcur | grep -v grep) ]]; do
    echo "[$i] Heart beat."
    sleep $sleep
    i=$(($i + 1))
done

for expid in $(echo $run | tr ':' '\n'); do

    bash exp/$expid/$script --sta 3 || exit 1
done
echo "Done."
