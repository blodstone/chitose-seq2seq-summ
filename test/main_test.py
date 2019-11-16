import unittest

import torch
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from main import SummDataReader


class MyTestCase(unittest.TestCase):
    def test_vocabulary_on_two_fields(self):
        tokenizer = WordTokenizer()
        indexer = {'single_id_indexer': SingleIdTokenIndexer('train')}
        reader = SummDataReader(tokenizer, indexer)
        sample_dataset = reader.read_data('../data/sample/src.txt', '../data/sample/tgt.txt')
        vocab = Vocabulary.from_instances(sample_dataset)
        self.assertEqual(6+2, vocab.get_vocab_size('train'))


if __name__ == '__main__':
    unittest.main()
