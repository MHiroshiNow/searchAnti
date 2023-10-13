import tensorflow as tf
print(tf.__version__)

import MeCab
import numpy as np
import pandas as pd
import gensim
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.model_selection import LeaveOneOut, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier








def create_model(input_length, vocab_size, embedding_dim, weights):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, weights=[weights], trainable=False),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # 形態素解析器の初期化
    mecab = MeCab.Tagger("-Owakati")

    # データの読み込み
    df = pd.read_csv('comment.csv', encoding='utf-8')

    # コメントとラベルの抽出
    texts = df['コメント'].values
    labels = df['ラベル'].values

    # テキストを単語に分割
    texts = [mecab.parse(str(text)).strip() for text in texts]

    # テキストデータをトークン化し、シーケンスに変換
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post')

    # Save the tokenizer object
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Word2Vecモデルの読み込み
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('entity_vector.model.bin', binary=True)

    # モデルの定義
    keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=50, verbose=0, input_length=padded_sequences.shape[1], vocab_size=len(word2vec.index_to_key), embedding_dim=word2vec.vector_size, weights=word2vec.vectors)

    # LOOCVを行う
    loo = LeaveOneOut()
    scores = cross_val_score(keras_model, padded_sequences, labels, cv=loo)

    # 平均スコアを出力
    print('Average score: ', scores.mean())

    # モデルの保存
save_model(keras_model, 'model.h5')


if __name__ == "__main__":
    main()
