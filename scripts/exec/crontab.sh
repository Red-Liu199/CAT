pidcur=$(ps aux | grep run.sh | grep -v grep | grep zhenghh | awk '{print $2}')

while [[ $(ps aux | grep $pidcur | grep -v grep) ]]; do
    echo "Heart beat."
    sleep 300
done

for id in 16; do
    bash exp/rnnt-v$id/run.sh
    echo $id
done
echo "Done."
