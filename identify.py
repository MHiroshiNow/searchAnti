import MeCab
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    # 形態素解析器の初期化
    mecab = MeCab.Tagger("-Owakati")

    # ユーザーからの入力を受け取る
    comment = input('Please enter a comment: ')

    # コメントを単語に分割
    comment = mecab.parse(comment).strip()

    # Load the tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # コメントをシーケンスに変換
    sequence = tokenizer.texts_to_sequences([comment])

    # シーケンスをパディング
    padded_sequence = pad_sequences(sequence, maxlen=11, padding='post')  # Change the maxlen to the length of sequences used in training

    # モデルの読み込み
    model = load_model('model.h5')


    # 予測の実行
    prediction = model.predict(padded_sequence)

    # 予測結果の表示
    if prediction > 0.5:
        print('This comment is classified as bad.')
        
    else:
        print('This comment is classified as good.')
        
        
    print(prediction)
if __name__ == "__main__":
    main()
