for i in 1 2 3 4 5
do
    python -u CLCD_mwe.py --src en-books --trg fr-dvd fr-music de-dvd de-music --epochs 20000 --export export/run_CLCD/
    python -u CLCD_mwe.py --src en-dvd --trg fr-books fr-music de-books de-music --epochs 20000 --export export/run_CLCD/
    python -u CLCD_mwe.py --src en-music --trg fr-books fr-dvd de-books de-dvd --epochs 20000 --export export/run_CLCD/
done
