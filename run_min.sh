for i in 1 2 3
do
    for lang in ja fr de
    do
        for dom in books dvd music
        do
            python -u slsd_senti.py --lang en $lang --trg $lang --val_interval 2000 --epochs 35000 --dom $dom --sup_dom $dom --lambd_lang 0.1 --lambd_dom 0 --dis_nhid 400 --export export/d-$lang-$dom-$i
        done
    done
done
