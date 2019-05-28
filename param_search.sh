args=$@

for seed in 0 1 2 3 4
do
    for lambd in 0.003 0.01 0.03 0.1
    do
        python -u cross_lingual_in_domain.py --lambd_clf $lambd \
               --src en --trg ja --lang en ja --dom books --sup_dom books --max_steps 50000 \
               --export export/param_search/clid-min-en-ja-books-lambd-$lambd-seed-$seed/ --seed $seed $args
    done
done
