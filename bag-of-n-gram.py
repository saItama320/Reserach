#!/usr/bin/env python
# coding: utf-8

# Purpose: Wordfile,Wordlistを走査して、頻度を元にn-gramの特徴量を作成(Make_feature.pyでWordfile, Wordlistを作成)
# Input: n-gramのn, 正のwordfile, 負のwordfile, wordlist
# Output: 頻度による特徴量ベクトルファイル

import sys
import pandas as pd
from itertools import product
import os
import csv
from tqdm import tqdm
import re

# なかった場合、新たにディレクトリを作成
def make_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

#ngram作成時のベクトルの特徴量を作成。つまり全単語リストからngramを作成
# n: n-gramのn
# words: wordlist
def Make_ngram_features(n, words):
    features = [0] * (len(words) ** n)
    for i ,l in enumerate(product(words,repeat=n)): # product : 直積
        features[i] = l[0] 
        for j in range(n-1):
            features[i] += "/" + l[j+1]
    return features

#単語が順番に入ったリストをngramのリストにする関数の定義
# n: n-gramのn
# words: 単語
def Make_ngram(n,words):#引数はngramのnと単語が入ったリスト
    words_ngram = [0]*(len(words)-(n-1))
    for i in range(len(words_ngram)):
        words_ngram[i] = words[i] 
        for j in range(n-1):
            words_ngram[i] += "/" + words[i+j+1]
    return words_ngram

#全wordファイルを走査してvecファイルに書き込む
#n: n-gramのn
#output_file: 出力先ファイル
#wordlist_features: wordlistのヘッダー名
#wordlist_dimension: 単語数
#wordfile: wordfile
def Make_vec_file(n, output_file, wordlist_features, wordlist_dimension, wordfile):

    with open(output_file, mode='w',newline="") as csvf:
        writer = csv.writer(csvf)
        #"ID"とベクトルの特徴量(単語)をcsvファイルの1行目に書き出す
        id_and_wordlist= [0] * 1
        id_and_wordlist[0] = "ID"
        id_and_wordlist.extend(wordlist_features)
        writer.writerow(id_and_wordlist)

        #全wordファイルを走査
        with open(wordfile, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = []
            id_list = []
            for row in reader:
                id_list.append(row[0])
                data.append(row[1:])
            del id_list[0]
            del data[0]      
            
        # 1検体ずつ見ていく
        for i in tqdm(range(len(data)), desc="loop:", leave=False):
            vec = [0] * wordlist_dimension
            #wordfileをwordfile_ngramに変換(単語をngram化)
            wordfile_ngram = Make_ngram(n, data[i])
            #countメソッドで各要素の出現頻度を求め、全単語リストに合わせたベクトル(リスト)を作成
            for j in range(wordlist_dimension):
                vec[j] = wordfile_ngram.count(wordlist_features[j])
                
            # 以下、実際にファイルに出力していく
            #idとベクトルの両方を含むリストを作成
            id_and_vec = []
            id_and_vec.append(id_list[i])#idをリストの先頭にする                                                        
            id_and_vec.extend(vec)#appendではなくextend．appendだと2次元配列になる．
                
            #ベクトルをcsvファイルに書き出す
            writer.writerow(id_and_vec)


def main():
    n = int(sys.argv[1])#n-gramのn#int()がないと文字列として入力されるので注意
    positive_word = sys.argv[2]
    negative_word = sys.argv[3]
    wordlist_path = sys.argv[4]
    
    text_ver_name = re.search('\d+', positive_word).group()

    # 全単語リストファイルをリストとして読み込む
    wordlist_df = pd.read_csv(wordlist_path)
    wordlist = wordlist_df["word"].values.tolist()

    # n-gramのnによる特徴量ベクトルを作成
    wordlist_features = Make_ngram_features(n,wordlist)
    wordlist_dimension = len(wordlist_features)#全単語リストのngram単語数。

    # 保存するディレクトリの作成
    vec_dir_path = "Vector/"
    make_dir(vec_dir_path)
    vec_dir_path += str(n) + "gram/"
    make_dir(vec_dir_path)
    outputfile_positive = vec_dir_path + str(n) + "gram_" + text_ver_name + "positive.csv"
    outputfile_negative = vec_dir_path + str(n) + "gram_" + text_ver_name + "negative.csv"
    
    Make_vec_file(n, outputfile_positive, wordlist_features, wordlist_dimension, positive_word)
    Make_vec_file(n, outputfile_negative, wordlist_features, wordlist_dimension, negative_word)
    
if __name__ == "__main__":
    main()
