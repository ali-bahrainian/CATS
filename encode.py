import struct
import tensorflow as tf
from tensorflow.core.example import example_pb2
import stanza
import os
from pathlib import Path

def tokenize_input(input_text):
    """Maps input text to a tokenized version using Stanford CoreNLP Tokenizer"""
    stanza.download('en')

    nlp = stanza.Pipeline('en', processors='tokenize')
    doc = nlp(input_text)
    
    tokenized_text = '\n'.join([' '.join([token.text for token in sentence.tokens]) for sentence in doc.sentences])
    return tokenized_text

def write_to_bin(input_text, out_file):
    """Read the input text, tokenize it, and write it to an out_file"""
    tokenized_input = tokenize_input(input_text)

    # Create directory for specific input file
    Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)

    with open(out_file, 'wb') as writer:
        # Write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([tokenized_input.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([''.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))