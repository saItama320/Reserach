# -*- condig: utf-8 -*-

# Purpose: 二値分類を行う(RF,DT)
# Input: 正ベクトル、正ベクトルの内学習する検体数、正ベクトルの内テストする検体数、負ベクトル、負ベクトルの内学習する検体数、負ベクトルの内テストする検体数、モデル名(RF or DT)

import os
import csv
import pickle
import numpy as np
import xgboost as xgb
import datetime
import pandas as pd
import sys
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ランダムに数字を選ぶ
def Random_num(positive_train_num, positive_test_num):
    num = list(range(1,101))
    random_num = random.sample(num, positive_train_num+positive_test_num)
    print(random_num)
    return random_num

# 正負検体をシャッフル
def shuffle(data, label):
    l = list(zip(data, label))
    np.random.shuffle(l)
    shuffle_data, shuffle_label = zip(*l)
    shuffle_data = pd.DataFrame(shuffle_data)
    shuffle_label = pd.DataFrame(shuffle_label)
    return shuffle_data, shuffle_label

# ラベルを作成
def Make_label(positive_num, negative_num):
    label = []
    for i in range(positive_num):
        label.append(0)
    for i in range(negative_num):
        label.append(1)
    return label

#データ事前準備
def pre(positive_vec, negative_vec, positive_train_num,  positive_test_num, negative_train_num, negative_test_num, random_num):
    # ファイルを読み込み
    df_positiveFile = pd.read_csv(positive_vec, header=None, skiprows=lambda x: x not in random_num, encoding="utf-8")
    #print(f'df_positiveFile: ',df_positiveFile)
    df_negativeFile = pd.read_csv(negative_vec, header=None, skiprows=lambda x: x not in random_num, encoding="utf-8")
    #print(f'df_negativeFile: ', df_positiveFile)

    columns = df_positiveFile.columns.values[0:] # ヘッダーを抽出
    #print(f'columns: ',columns)

    # ランダムに抽出したうちの、最初の数個はtrain、残りをtestにする
    extract_train_positiveFile = df_positiveFile.iloc[:positive_train_num, :]
    extract_test_positiveFile = df_positiveFile.iloc[positive_train_num:, :]       
    extract_train_negativeFile = df_negativeFile.iloc[:negative_train_num, :]
    extract_test_negativeFile = df_negativeFile.iloc[negative_train_num:, :]       
    #print(f'extract_train_negativeFile: ',extract_train_negativeFile)
    #print(f'extract_test_negativeFile: ', extract_test_negativeFile)

    # 結合
    extract_train = np.array(pd.concat([extract_train_positiveFile, extract_train_negativeFile]))
    extract_test = np.array(pd.concat([extract_test_positiveFile, extract_test_negativeFile]))
    np.delete(extract_train, 0, 1)
    np.delete(extract_test, 0, 1)

    # id_list
    id_train_list = extract_train_positiveFile.iloc[:,0].values
    id_train_list = np.concatenate([id_train_list, extract_train_negativeFile.iloc[:,0].values])
    id_test_list = extract_test_positiveFile.iloc[:,0].values
    id_test_list = np.concatenate([id_test_list, extract_test_negativeFile.iloc[:,0].values])
    #print(f'id_test_list: ',id_test_list)
  
    label_train = Make_label(positive_train_num, negative_train_num)
    label_test = Make_label(positive_test_num, negative_test_num)

    # シャッフル
    shuffle_train_data, shuffle_train_label, shuffle_id_train = shuffle(np.array(extract_train), label_train, id_train_list)
    shuffle_test_data, shuffle_test_label, shuffle_id_test = shuffle(np.array(extract_test), label_test, id_test_list)
    #print(f'shuffle_test_data: ', shuffle_train_data)
    #print(f'shuffle_train_label: ', shuffle_train_label)

    return shuffle_train_data, shuffle_test_data, shuffle_train_label, shuffle_test_label, shuffle_id_train, shuffle_id_test

# 分類
def predict_process(X_train, X_test, y_train, y_test, model_name, id_list, dir_path):
    # 混同行列の初期化
    cm = np.zeros((2,2))
    
    result_classification = []# 検体ごとの分類結果を保存するリスト
    feature_importances = [0]* X_train.shape[1]# k分割ごとの特徴量の寄与率を保存するリスト

    """
    # 特徴量選択を行う場合の、outputファイルの準備
    output_file_used_features = dir_path + "used_features.csv"
    writeCSV(output_file_used_features, ["======="])
    writeCSV(output_file_used_features, [datetime.datetime.now()]) # 現在時刻"""
       
    # データの標準化処理# 平均0，分散1にする
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)#fitはしない．これにより学習データを標準化したときの平均と標準偏差を利用できる． # 訓練データは学習データで用いた変換式でfitする必要があるため
            
    # モデルのインスタンスを生成．パラメータ候補を設定
    if model_name == "RF":
        model = RandomForestClassifier()
        candidate_params = {
            "n_estimators":[100],#100がデフォルト
            #"n_estimators":[1],
            "criterion":["gini","entropy"],
            #"max_depth":[3,4,5,6,7,8,9],#[i for i in range(3,12,2)],#1,3...,11
            "max_depth":[6,7,8,9,10],#[i for i in range(3,12,2)],#1,3...,11
            #"max_features":[10,11,12,13,14],
            #"max_features":[15],
            "random_state":[0],
        }
    elif model_name == "DT":
        model = DecisionTreeClassifier()
        candidate_params = {
            "criterion":["gini","entropy"],
            #"criterion":["gini"],
            #"max_depth":[5,7,9],#[i for i in range(3,12,2)],#1,3...,11
            "max_depth":[7,8,9,10,11],#[i for i in range(3,12,2)],#1,3...,11
            #"max_depth":[1],
            "random_state":[0],
        }

    # GridSearchCVでパラメータを決定        
    skf_para = StratifiedKFold(n_splits=3, shuffle=False)#,random_state=0)#sfkインスタンス生成 # trainとtestに入っているラベルの比率を合わせてくれる
    gs = GridSearchCV(estimator=model, param_grid=candidate_params,scoring="accuracy", cv=skf_para, n_jobs=os.cpu_count()) # パラメータを全部試してスコアを返してくれる            
    gs.fit(X_train_std, y_train)
            
    # 最適なパラメータを登録しておく
    best_model = gs.best_estimator_ # estimator:推定器
        
    # 最適なパラメータを見つけられたので、そのパラメータでSelectFromModelを用いて特徴量選択
    # その後、その特徴量に対して再度最適なパラメータを決定
    restrict_num = 1000
    selector = SelectFromModel(best_model, threshold = "mean", max_features = restrict_num) # thresholdは閾値。meanはデフォルトで、中央値。max_featuresは最大特徴量数
    selector.fit(X_train_std, y_train)
    mask = selector.get_support() # その特徴量が使用されているか否かを示す
        
    # 選択した特徴量のみを残したベクトル
    X_train_std_trans = selector.transform(X_train_std)
    X_test_std_trans = selector.transform(X_test_std)
        
    # 抽出した特徴量に対して最適なパラメータを決定  
    skf_para = StratifiedKFold(n_splits=3, shuffle=False)#,random_state=0)#sfkインスタンス生成 # trainとtestに入っているラベルの比率を合わせてくれる
    gs = GridSearchCV(estimator=model, param_grid=candidate_params,scoring="accuracy", cv=skf_para, n_jobs=os.cpu_count()) # パラメータを全部試してスコアを返してくれる            
    gs.fit(X_train_std_trans, y_train)
    # 最適なパラメータを更新
    best_model = gs.best_estimator_
            
    # テストデータを予測
    # 再度、特徴量選択後のベクトルでfitさせる
    best_model.fit(X_train_std_trans, y_train)
    y_pred = best_model.predict(X_test_std_trans)
    
    # 混同行列に予測結果を追加
    for i in range(y_pred):
        cm[y_test[i]][y_pred[i]] += 1
        if y_test[i] == y_pred[i]:
            flag = 1
        else:
            flag = 0
        result_classification.append([id_list[i],y_test[i],y_pred[i],flag])#[テストデータの検体ID,正解ラベル，予測ラベル，正解フラグ ]
                                 
    # ベストパラメータ
    best_para = gs.best_params_
            
    # accuracyの計算
    correct_sum = 0
    for i in range(2):
        correct_sum += cm[i][i] 
    cm_sum = np.array(cm).sum()#リストのcmをndarrayにしてsumメソッドで要素の合計値を求める
    accuracy = correct_sum / cm_sum
    print("accuracy : ", accuracy)
    
    '''
    # 重要度の和を、大きい順にソートしたもののインデックスを抽出
    sort_index_feature_importances = np.argsort(feature_importances)[::-1]
    
    # 上位100個だけファイルで保存する
    output_file_feature_importances = dir_path + "feature_importances_" + str(os.path.basename(vecFile_path))
    writeCSV(output_file_feature_importances, ["======"])
    writeCSV(output_file_feature_importances, [datetime.datetime.now()]) # 現地時刻取得
    for i in range(100):
        try:
            writeCSV(output_file_feature_importances, [i+1, column[sort_index_feature_importances[i]], feature_importances[sort_index_feature_importances[i]]])
        except IndexError as e:
            break
    writeCSV(output_file_feature_importances, ["======"])'''
        
    return accuracy, cm , result_classification, candidate_params

def main():
    positive_vec = sys.argv[1] #正ベクトル
    positive_train_num = int(sys.argv[2]) #正ベクトルの内学習する検体数
    positive_test_num = int(sys.argv[3]) #正ベクトルの内テストする検体数
    negative_vec = sys.argv[4] #負ベクトル
    negative_train_num = int(sys.argv[5]) #負ベクトルの内学習する検体数
    negative_test_num = int(sys.argv[6]) #負ベクトルの内テストする検体数
    model = sys.argv[7]# モデル名(RF or DT)
    
    #CSVファイルのdataframeへの読み込み
    num = []
    random_num = Random_num(positive_train_num, positive_test_num) # ランダムに検体数を選ぶ
    print(random_num)
    
    # データ事前準備
    shuffle_train_data, shuffle_test_data, shuffle_train_label, shuffle_test_label, shuffle_id_train, shuffle_id_test = pre(
        positive_vec, negative_vec, positive_train_num,  positive_test_num, negative_train_num, negative_test_num, random_num)
    
    output_dir = "Predict/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # 学習
    accuracy, cm, result_classification, candidate_params = predict_process(shuffle_train_data, shuffle_test_data, shuffle_train_label, shuffle_test_label, model, shuffle_id_test, output_dir)
    
if __name__ == "__main__":
    main()
