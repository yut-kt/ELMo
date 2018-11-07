# -*- coding: utf-8 -*-

import os
import MeCab
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def main():
    tags, sentences = get_tuple_tags_sentences()
    sentences = get_word_strList(sentences)
    features = None
    with tf.Session() as sess:
        for sliced_tags, sliced_sentences, in zip(slice_list(tags, 100), slice_list(sentences, 100)):
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embeddings = elmo(sliced_sentences, signature='default', as_dict=True)['elmo']
            if features is None:
                features = sess.run(embeddings).sum(axis=1)
            else:
                features = np.r_[features, sess.run(embeddings).sum(axis=1)]
    np.savez(f'./data/feature.npz', elmo=features, labels=tags)


def get_tuple_tags_sentences():
    """
    ファイルを読み込んでタグと文に分割
    :return: タグ配列と文配列のタプル
    """
    tags, sentences = [], []
    with open(train_file) as fp:
        for line in fp:
            tag, _, sentence = line.strip().split(maxsplit=2)
            tags.append(tag)
            sentences.append(sentence)
    return tags, sentences


def get_word_strList(sentences):
    def get_word_str(sentence) -> str:
        def validate(word_line):
            if word_line.strip() == 'EOS' or word_line == '':
                return
            word, info = word_line.split('\t')
            part, fine_part, _ = info.split(',', maxsplit=2)
            if part == '記号' or fine_part == '数':
                return
            return word

        return ' '.join(filter(None, [validate(word_line) for word_line in mecab.parse(sentence).split('\n')]))

    return [get_word_str(sentence) for sentence in sentences]


def slice_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return:
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


if __name__ == '__main__':
    train_file = './data/train.list'
    mecab = MeCab.Tagger()

    os.environ['TFHUB_CACHE_DIR'] = './.module_cache'
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    main()
