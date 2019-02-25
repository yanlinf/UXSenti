import MeCab
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import os

SRC_DIR = 'cls-acl10-unprocessed/'
TRG_DIR = '../data/'
LANGS = ['en', 'fr', 'de', 'jp']
DOMAINS = ['books', 'dvd', 'music']
PART = ['train.review', 'test.review', 'unlabeled.review']


def create(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(path, 'w', encoding='utf-8') as fout:
        pass


class Tokenizer(object):
    """
    a tokenizer for multiple languages (western languages and Japanese supported)
    """

    def __init__(self, lang, lower_case=True, num_token='<num>'):
        self.lang = lang
        self.lower_case = lower_case
        self.num_token = num_token

        if self.lang == 'jp':
            self.tagger = MeCab.Tagger('-Ochasen')

    def tokenize(self, text):
        if self.lang == 'jp':
            ans = self.tagger.parse(text)
            tokens = [t.split('\t')[0] for t in ans.split('\n')][:-2]
        else:
            tokens = word_tokenize(text)

        # do lower casing
        if self.lower_case:
            tokens = [t.lower() for t in tokens]

        # converting all-digit tokens to special tokens
        if self.num_token is not None:
            tokens = [self.num_token if t.isdigit() else t for t in tokens]

        return tokens


def main():
    # testing
    """
    jp_tagger = Tokenizer('jp', lower_case=True)
    tagger = Tokenizer('en', lower_case=True)
    print(jp_tagger.tokenize("ベスト版を引っさげてなので，有名な曲ばかりだと思い込んで買うのをためらってました。しかし，よく見ると他のライブで演奏されにくい曲が揃っていて充実してました。ROUNDABOUTとかLOVEがすごく良かったです。しかし，有名な曲「ニシエヒガシエ，TOMORROWNEVERKNOWS」が迫力に欠けていたのが残念。"))
    print(jp_tagger.tokenize('''１９８８年の公開以来、２０年以上たった今、初めてこの映画を見た。
 「ＭＡＮ　ＩＮ　ＴＨＥ　ＭＩＲＲＯＲ」のライヴ映像に始まり、彼のジャクソン５時代から１９８８年当時までの活躍を振り返る映像、「ＳＰＥＥＤ　ＤＥＭＯＮ」のショートフィルムと続く。さらに「ＬＥＡＶＥ　ＭＥ　ＡＬＯＮＥ」のショートフィルム。そして次に、やっとこの『ＭＯＯＮ　ＷＡＬＫＥＲ』という映画の本編とでも言うべきものが始まる。「ＳＭＯＯＴＨ　ＣＲＩＭＩＮＡＬ」のショートフィルムが入る部分だ。世界をドラッグ漬けにして、ダメにしてしまおうと企む組織に、マイケルは一人で立ち向かい、自ら巨大な戦闘型ロボットに変身し、一味を退治するというものだ。捕らえられていた女の子を助け出し、自分は宇宙に帰っていき、物語は終了。かと思いきや、マイケルは再び姿を現す。そして「さあ、行こう。」と３人の子供達を誘って、コンサート会場へ。そこで彼は、ビートルズの「ＣＯＭＥ　ＴＯＧＨＥＴＨＥＲ」を歌うのだ。そこで映画は終了。そして――その次である。「ＳＭＯＯＴＨ　ＣＲＩＭＩＮＡＬ」の舞台に、帽子をかぶり背広に身を包んだ黒人のおじさんたちが１０人ほど現れ、黒人霊歌のようなものを歌い、輪になって民族舞踊みたいなものを踊り出したのだ。そこに、この映画のクレジットがかぶせられる。'''))
    print(tagger.tokenize("I grew up watching this and just love watching the show now as an adult!  I really love that these came out on DVD and wonder when season 4 is coming out? Hurry!  :)  It's nice to have a show that's clean, interesting, has intelligent characters, and entertaining at the same time!  Woo hoo"))
    exit(9)
    """

    for lang in LANGS:
        tokenizer = Tokenizer(lang, lower_case=True, num_token='<num>')
        trg_lang_file = os.path.join(TRG_DIR, lang, 'full.review')
        create(trg_lang_file)

        for dom in DOMAINS:
            trg_lang_dom_file = os.path.join(TRG_DIR, lang, dom, 'full.review')
            create(trg_lang_dom_file)

            for part in PART:
                trg_file = os.path.join(TRG_DIR, lang, dom, part)
                create(trg_file)

                root = ET.parse(os.path.join(SRC_DIR, lang, dom, part)).getroot()
                nitem, npos, nneg = 0, 0, 0
                # processed = []
                for t in root:
                    try:
                        dic = {x.tag: x.text for x in t}
                        if part == 'unlabeled.review':
                            label = '__unk__'
                        else:
                            label = '__pos__' if float(dic['rating']) > 3 else '__neg__'

                        tokens = tokenizer.tokenize(dic['text'])
                        # processed.append(label + ' ' + ' '.join(tokens))
                        with open(trg_file, 'a', encoding='utf-8') as fout:
                            fout.write(label + ' ' + ' '.join(tokens) + '\n')

                        if part != 'test.review':
                            with open(trg_lang_file, 'a', encoding='utf-8') as fout:
                                fout.write(' '.join(tokens) + '\n')

                            with open(trg_lang_dom_file, 'a', encoding='utf-8') as fout:
                                fout.write(' '.join(tokens) + '\n')

                        if label == '__pos__':
                            npos += 1
                        elif label == '__neg__':
                            nneg += 1
                        nitem += 1

                    except Exception as e:
                        print('[ERROR] ignoring item - {}'.format(e))

                print('file: {}   valid: {}   pos: {}   neg: {}'.format(os.path.join(lang, dom, part), nitem, npos, nneg))


if __name__ == '__main__':
    main()
