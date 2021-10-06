# Reserach

研究の過程で、勉強したことを経て作成したコードである。

## 想定使用データ

Tweetを集めたtxtファイル 200検体(男女100検体ずつ)

## 想定状況
全てのデータを使用するのではなく、一部データをランダムに抽出して使用する(使用検体数をあえて絞る)

## 使い方

1. make_word_used_mecab.py: txtファイルからで形態素解析(出力：wordfile,wordlist)
2. bag-of-ngram.py: wordfile, wordlistを使用して、各単語の出現頻度による特徴量ベクトルを作成
3. classificaation.py: ランダムに一部データを取り出して、二値分類
