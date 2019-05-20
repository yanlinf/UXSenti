: ${1?"Usage: $0 SEED"}
seed=$1; shift
args=$@

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg de-books --export export/clcd-en-d-de-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg de-books --export export/clcd-en-m-de-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg de-dvd --export export/clcd-en-b-de-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg de-dvd --export export/clcd-en-m-de-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg de-music --export export/clcd-en-b-de-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg de-music --export export/clcd-en-d-de-m/ $args

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg fr-books --export export/clcd-en-d-fr-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg fr-books --export export/clcd-en-m-fr-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg fr-dvd --export export/clcd-en-b-fr-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg fr-dvd --export export/clcd-en-m-fr-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg fr-music --export export/clcd-en-b-fr-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg fr-music --export export/clcd-en-d-fr-m/ $args

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg ja-books --export export/clcd-en-d-ja-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg ja-books --export export/clcd-en-m-ja-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg ja-dvd --export export/clcd-en-b-ja-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-music --trg ja-dvd --export export/clcd-en-m-ja-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-books --trg ja-music --export export/clcd-en-b-ja-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 30000 --src en-dvd --trg ja-music --export export/clcd-en-d-ja-m/ $args

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg de-books --export export/clcd-mwe-en-d-de-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg de-books --export export/clcd-mwe-en-m-de-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg de-dvd --export export/clcd-mwe-en-b-de-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg de-dvd --export export/clcd-mwe-en-m-de-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg de-music --export export/clcd-mwe-en-b-de-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg de-music --export export/clcd-mwe-en-d-de-m/ $args

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg fr-books --export export/clcd-mwe-en-d-fr-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg fr-books --export export/clcd-mwe-en-m-fr-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg fr-dvd --export export/clcd-mwe-en-b-fr-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg fr-dvd --export export/clcd-mwe-en-m-fr-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg fr-music --export export/clcd-mwe-en-b-fr-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg fr-music --export export/clcd-mwe-en-d-fr-m/ $args

python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg ja-books --export export/clcd-mwe-en-d-ja-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg ja-books --export export/clcd-mwe-en-m-ja-b/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg ja-dvd --export export/clcd-mwe-en-b-ja-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-music --trg ja-dvd --export export/clcd-mwe-en-m-ja-d/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-books --trg ja-music --export export/clcd-mwe-en-b-ja-m/ $args
python -u cross_lingual_cross_domain.py --seed $seed --max_steps 20000 --mwe --src en-dvd --trg ja-music --export export/clcd-mwe-en-d-ja-m/ $args
