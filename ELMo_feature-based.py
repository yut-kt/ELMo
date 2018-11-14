# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import os
import MeCab
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def main():
    tags, sentences = get_tuple_tags_sentences()
    sentences = get_word_strList(sentences)
    features = None
    max_size = len(tags)
    with tf.Session() as sess:
        for index, sliced_sentences in enumerate(slice_list(sentences, args.num_split_elements)):
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embeddings = elmo(sliced_sentences, signature='default', as_dict=True)['elmo']
            if features is None:
                features = sess.run(embeddings)
            else:
                features = np.r_[features, sess.run(embeddings)]

            size = (index + 1) * args.num_split_elements
            if size > max_size:
                size = max_size
            print(size, '/', max_size, flush=True)

    np.savez(f'{args.output_name}.npz', elmo=features, labels=tags)


def get_tuple_tags_sentences():
    """
    ファイルを読み込んでタグと文に分割
    :return: タグ配列と文配列のタプル
    """
    tags, sentences = [], []
    with open(args.input_file) as fp:
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
    parser = ArgumentParser(description='ELMo Feature Based')
    parser.add_argument('-i', '--input_file', help='入力ファイルパス', required=True)
    parser.add_argument('-o', '--output_name', help='出力ファイル名', required=True)
    parser.add_argument('-c', '--cache_dir', help='ELMoのダウンロードパス', default='.module_cache')
    parser.add_argument('-n', '--num_split_elements', help='ELMo1回にかける数', type=int, default=100)
    args = parser.parse_args()

    mecab = MeCab.Tagger()

    os.environ['TFHUB_CACHE_DIR'] = args.cache_dir
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    main()
