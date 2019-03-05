for i in 1 2 3
do
    for lang in ja fr de
    do
        for dom in books dvd music
        do
            python -u slsd_senti.py --lang en $lang --trg $lang --val_interval 2000 --epochs 50000 --dom $dom --sup_dom $dom --export export/min-$lang-$dom-$i/
        done
    done
done
