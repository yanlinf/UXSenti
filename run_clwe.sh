for i in 1 2 3
do
    for dom in books dvd music
    do
        python -u mwe.py --sup_dom $dom --epochs 20000 --export export/clwe-$dom-$i/
    done
done
