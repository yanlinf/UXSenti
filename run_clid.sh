: ${1?"Usage: $0 SEED"}
seed=$1; shift
args=$@

# run CLIDSA_{full}
for dom in books dvd music
do
    python -u cross_lingual_in_domain.py --seed $seed --max_steps 50000 \
           --src en --trg fr de ja --sup_dom $dom --export export/clid-full-$dom/ $args
done

# run CLIDSA_{min}
for dom in books dvd music
do
    for lang in fr de ja
    do
        python -u cross_lingual_in_domain.py --seed $seed --max_steps 50000 \
               --lang en $lang --dom $dom --src en --trg $lang --sup_dom $dom \
               --export export/clid-min-en-$lang-$dom/ $args
    done
done

# run MWE
for dom in books dvd music
do
    python -u cross_lingual_in_domain.py --seed $seed --max_steps 20000 \
           --mwe --src en --trg fr de ja --sup_dom $dom --export export/clid-mwe-$dom/ $args
done
