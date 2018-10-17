max=5
for i in `seq 1 $max`
do
    echo "$i"
    for j in `seq 1 $max`
    do
        echo "$j"
        KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/imdb.npz --db imdb --netType1 $i --netType2 $j
        KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/wiki.npz --db wiki --netType1 $i --netType2 $j --batch_size 32
        #KERAS_BACKEND=tensorflow python SSRNET_train.py --input ../data/morph2_db_align.npz --db morph --netType1 $i --netType2 $j --batch_size 50
    done
done
