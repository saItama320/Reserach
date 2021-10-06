# coding:utf-8

# Purpose: Mecabでtxtファイルを形態素解析
# Input: txtファイルがあるディレクトリ(正負の2種類)
# Output: 形態素解析で得られた各特徴量(Wordfile)、全ファイル中で出現する特徴量の種類(Wordlist)

import MeCab
import unidic
import sys
from tqdm import tqdm
from pathlib import Path
import csv
import os
import re
import pandas as pd
import numpy as np

# 形態素解析を行う関数
# output_file:書き込み先ファイル名
# file_list:解析対象ファイルのリスト
def mecab_extract(output_file, file_list):

    # 書き込み先パス
    with open(output_file, mode='w', newline="") as fw:
        writer = csv.writer(fw)
        header = ["ID", "word"]
        writer.writerow(header)
        
        # ファイルを開いて読み込む
        for file in tqdm(file_list, desc='loop: ', leave=False):
            id = re.findall(r'\d+',str(file))[-1]

            fr = open(file, 'r', encoding='UTF-8')
            data = fr.read()        
            tagger = MeCab.Tagger()
            result = tagger.parse(data)
            
            word_list = [id]
            # 形態素解析のあとの単語だけ抽出
            for text in result.split("\n"):
                word = text.split("\t")[0]
                if word != "EOS":
                    word_list.append(word)
                else:
                    break
            writer.writerow(word_list)
            fr.close()            


# 正負二つで行う
def Make_word(wordfile_positive_path, wordfile_negative_path, positive_dir_list, negative_dir_list):
    
    mecab_extract(wordfile_positive_path, positive_dir_list)
    mecab_extract(wordfile_negative_path, negative_dir_list)
    
    
# 全ファイル中で出現する特徴量の種類を出力したWordlistを作成する
# output_file:書き込み先ファイル
# positive_wordfile:正ファイル
# negative_wordfile:負ファイル
def Make_wordlist(output_file, positive_wordfile, negative_wordfile):
    # wordのみ抽出して、ユニークなものを探す
    with open(positive_wordfile, "r") as fp:
        po = csv.reader(fp, delimiter=",")
        data_positive = []
        for row in po:
            data_positive.extend(row[1:])
        del data_positive[0]
    
    with open(negative_wordfile, "r") as fn:    
        ne = csv.reader(fn, delimiter=",")
        data_negative = []
        for row in ne:
            data_negative.extend(row[1:])
        del data_negative[0]
    df = pd.DataFrame(data_positive + data_negative)

    # uniqueな単語と出現頻度を抽出
    vc = pd.DataFrame(df.value_counts())
    vc.columns = [1] # column, indexがともに0なので、一旦変えないとリセットできない

    # csvに書き込む前にヘッダーをつける
    vc_df = vc.reset_index()#ヘッダーを追加するためデータフレームにし，インデックスをリセット
    vc_df.columns = ["word", "frequency"]
    vc_df.to_csv(output_file, index=False)
    
    


def main():
    # 入力
    text_dir_path = sys.argv[1]
    
    text_ver_name = re.search('\d+', text_dir_path).group()
    
    # ディレクトリ内のファイルパス取得
    all_positive_file_path_list = list(Path(text_dir_path+'/positive/tweet').iterdir())
    all_negative_file_path_list = list(Path(text_dir_path+'/negative/tweet').iterdir())
    
    wordfile_dir = "Wordfile/"
    if not os.path.exists(wordfile_dir):
        os.mkdir(wordfile_dir)     
    wordfile_positive_path = wordfile_dir + text_ver_name+ "_word_positive.csv"
    wordfile_negative_path = wordfile_dir + text_ver_name + "_word_negative_.csv"
    
    # wordファイル作成    
    Make_word(wordfile_positive_path, wordfile_negative_path, all_positive_file_path_list , all_negative_file_path_list)
    
    # wordリスト作成
    wordlist_dir = "Wordlist/"
    if not os.path.exists(wordlist_dir):
        os.mkdir(wordlist_dir)
    wordlist_file = wordlist_dir + "wordlist_" + text_ver_name + ".csv"
    
    # 全ファイル中で出現する特徴量の種類を出力したWordlistを作成する
    Make_wordlist(wordlist_file, wordfile_positive_path, wordfile_negative_path)

if __name__ == "__main__":
    main()
