python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-dvd --trg de-books --export export/cross-de/
python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-music --trg de-books --export export/cross-de/
python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-books --trg de-dvd --export export/cross-de/
python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-music --trg de-dvd --export export/cross-de/
python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-books --trg de-music --export export/cross-de/
python -u cross_lingual_cross_domain.py --epochs 30000 --lm_lr 0.003 --clf_lr 0.003 --dis_lr 0.003 --src en-dvd --trg de-music --export export/cross-de/
