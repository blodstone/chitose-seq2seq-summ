import os
from typing import Dict, Optional, Iterable

import torch
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import Model, ComposedSeq2Seq, SimpleSeq2Seq
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.seq2seq_decoders import SeqDecoder, DecoderNet, LstmCellDecoderNet, AutoRegressiveSeqDecoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import RegularizerApplicator, util
from allennlp.training import Trainer
from torch import optim


class SummDataReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 indexer: Dict[str, TokenIndexer] = None,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._indexer = indexer
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

    def read_data(self, src_file_path: str, tgt_file_path: str) -> Iterable[Instance]:
        src_file = open(src_file_path)
        tgt_file = open(tgt_file_path)
        for src_seq, tgt_seq in zip(src_file, tgt_file):
            yield self.text_to_instance(src_seq, tgt_seq)
        src_file.close()
        tgt_file.close()

    def text_to_instance(self, src_seq, tgt_seq) -> Instance:
        tokenized_src = self._tokenizer.tokenize(src_seq)[:self._source_max_tokens]
        tokenized_tgt = self._tokenizer.tokenize(tgt_seq)[:self._target_max_tokens]
        source_field = TextField(tokenized_src, self._indexer)
        target_field = TextField(tokenized_tgt, self._indexer)
        return Instance({'source_tokens': source_field, 'target_tokens': target_field})


class Seq2SeqModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 src_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: SeqDecoder,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self._src_embedder = src_embedder
        self._encoder = encoder
        self._decoder = decoder

    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.tensor]:
        state = self._encode(source_tokens)
        return self._decoder(state, target_tokens)

    def _encode(self, src_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_input = self._src_embedder(src_tokens)
        source_mask = util.get_text_field_mask(src_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
            'source_mask': source_mask,
            'encoder_outputs': encoder_outputs
        }


if __name__ == '__main__':
    vocab_path = 'data/cnndm/vocab'
    tokenizer = WordTokenizer(JustSpacesWordSplitter())
    indexer = {'tokens': SingleIdTokenIndexer('train')}
    reader = SummDataReader(tokenizer, indexer, source_max_tokens=400)
    train_dataset = reader.read_data(
        'data/cnndm/train.txt.src', 'data/cnndm/train.txt.tgt.tagged')
    validation_dataset = reader.read_data(
        'data/cnndm/val.txt.src', 'data/cnndm/val.txt.tgt.tagged')
    test_dataset = reader.read_data(
        'data/cnndm/test.txt.src', 'data/cnndm/test.txt.tgt.tagged')
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=80000)
        vocab.save_to_files(vocab_path)
    embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('train'),
        embedding_dim=128)
    embedder = BasicTextFieldEmbedder({'tokens': embedding})
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True))
    # decoder_net = LstmCellDecoderNet(decoding_dim=128, target_embedding_dim=128)
    # decoder = AutoRegressiveSeqDecoder(
    #     max_decoding_steps=100, target_namespace='train',
    #     target_embedder=embedding, beam_size=5, decoder_net=decoder_net, vocab=vocab)
    model = SimpleSeq2Seq(encoder=encoder, vocab=vocab, beam_size=5, max_decoding_steps=100, target_embedding_dim=128, source_embedder=embedder, target_namespace='train')
    # model = Seq2SeqModel(encoder=encoder, decoder=decoder, vocab=vocab, src_embedder=embedder)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=16, sorting_keys=[("source_tokens", "num_tokens")])
    iterator.index_with(vocab)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    trainer = Trainer(model=model, optimizer=optimizer, train_dataset=train_dataset, iterator=iterator, num_epochs=2, cuda_device=cuda_device)
    print('Begin Training')
    trainer.train()
