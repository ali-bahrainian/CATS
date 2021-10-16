# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import gensim
from gensim import corpora, similarities, models
import re

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)"""

  def __init__(self, vocab_file, max_size):
    """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
      max_size: integer. The maximum size of the resulting Vocabulary."""
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break

    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    """The topic model is initialized here"""
    self.tm = TopicModel()



  def word2id(self, word):
    """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer)."""
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    """Returns the total size of the vocabulary"""
    return self._count

  def write_metadata(self, fpath):
    """Writes metadata file for Tensorboard word embedding visualizer as described here:
      https://www.tensorflow.org/get_started/embedding_viz

    Args:
      fpath: place to write the metadata file
    """
    print("Writing word embedding metadata file to %s..." % (fpath))
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        writer.writerow({"word": self._id_to_word[i]})



class TopicModel (object):
    """Reads a pretrained LDA model trained using the Gensim library"""
    def __init__(self, modelAdd = 'lda/lda.model', dictAdd = 'lda/dictionary.dic', topicsFileAdd = ''):
        self.topicModel, self.topicModelDictionary = self._loadPretrainedTM (modelAdd, dictAdd)
        self.topics_dictionary = self._createTopicsDict(1000)# A dictionary containing all topics
    
    def _loadPretrainedTM (self, modelAdd, dictAdd):
        lda = gensim.models.ldamodel.LdaModel.load(modelAdd, mmap = 'r')
        print("Loaded the LDA model.")
        dictionary = corpora.Dictionary.load(dictAdd, mmap = 'r')
        print("Loaded dictionary.")
        return lda, dictionary
    
    def _doc2topics (self, doc):
        vec_bow = self.topicModelDictionary.doc2bow(doc.split())
        perDocTopics = self.topicModel[vec_bow]
        return perDocTopics
    
    def _turnTopicOff (self, topicDict):
        for k in list(topicDict.keys()):
            topicDict[k] = 0.0
        return topicDict

    def _createTopicsDict (self, numWordsPerTopic): # put all topics in a dictionary. A topic could be accessed by its number. Function returns a dictionary contating all topics.
        print("Creating a topic dictionary")
        topics_dictionary = {}
        tm = self.topicModel.show_topics(num_topics=150, num_words = numWordsPerTopic, formatted=False)
        for t in tm:
            print(t)
            temp = {}
            for tup in t[1]:
                temp[str(tup[0])] = tup[1]
                #"""The following if condition is for turning a topic off"""
                #if t[0]==3 or t[0]==17:
                #    temp = self._turnTopicOff(temp)
            topics_dictionary[t[0]]=temp
        print("Finished building a dictionary of all topics")
        return topics_dictionary
    
    
    """You should determine if topic proportions are taken into account using the Boolean parameter topicProportions"""
    def _doc2FinalWordVector (self, doc, topicProportions = True):
        docTopics = {}
        #Get the topics of the given doc
        topics = self._doc2topics(doc)
        print (topics)
        print((list(self.topics_dictionary.keys())))
        for tup in topics:
            for item in self.topics_dictionary[tup[0]].items():
                if item[0] in docTopics:
                    if topicProportions:
                        docTopics[item[0]]+=item[1]*tup[1]
                    else:
                        docTopics[item[0]]+=item[1]
                else:
                    if topicProportions:
                        docTopics[item[0]]=item[1]*tup[1]
                    else:
                        docTopics[item[0]]=item[1]
        return docTopics

    def get_doc_topics_words_probs (self, doc):
        docTopics = self._doc2FinalWordVector(doc, topicProportions=True)
        #words, prob_scores = docTopics.keys(), docTopics.values()
        words =[]
        prob_scores = []
        #put the words in the same order as the original document, for non-existing words use a very small probability e.g. 0.0001
        doc_words = doc.split(' ')
        for w in doc_words:
            words.append(w)
            if w in docTopics:
                prob_scores.append(docTopics[w])
            else:
                prob_scores.append(0.0)
        return words, prob_scores

def article_topicwords_2_ids(article_topic_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.
        
        Args:
        article_words: list of words (strings)
        vocab: Vocabulary object
        
        Returns:
        ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs:
        A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_topic_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs



def example_generator(data_path, single_pass):
  """Generates tf.Examples from data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path:
      Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
    single_pass:
      Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

  Yields:
    Deserialized tf.Example.
  """
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)
    if single_pass:
      print("example_generator completed reading all datafiles. No more data.")
      break


def article2ids(article_words, vocab):
  """Map the article words to their ids. Also return a list of OOVs in the article.

  Args:
    article_words: list of words (strings)
    vocab: Vocabulary object

  Returns:
    ids:
      A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
    oovs:
      A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
  """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

  Args:
    abstract_words: list of words (strings)
    vocab: Vocabulary object
    article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

  Returns:
    ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

  Args:
    id_list: list of ids (integers)
    vocab: Vocabulary object
    article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

  Returns:
    words: list of words (strings)
  """
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words


def abstract2sents(abstract):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  return [str.encode(sent) for sent in re.findall('<s> (.+?) <\/s>', abstract.decode('utf-8'))]


def show_art_oovs(article, vocab):
  """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
  """Returns the abstract string, highlighting the article OOVs with __underscores__.

  If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

  Args:
    abstract: string
    vocab: Vocabulary object
    article_oovs: list of words (strings), or None (in baseline mode)
  """
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if article_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str


