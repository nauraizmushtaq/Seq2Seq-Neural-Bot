#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing , utils
import pandas as pd


# ## Reading the data

# In[2]:


data_path = './clean_conversation.txt'


# In[3]:


input_texts = []
target_texts = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(9999, len(lines) - 1)]:
    input_text = line.split('\t')[0]
    target_text = line.split('\t')[1]
    input_texts.append(input_text)
    target_texts.append(target_text)


# In[4]:


len(input_texts)


# In[5]:


zippedList =  list(zip(input_texts, target_texts))
lines = pd.DataFrame(zippedList, columns = ['input' , 'output']) 


# In[6]:


lines.head()


# ## Preparing input data for the Encoder

# In[7]:


input_lines = list()
for line in lines.input:
    input_lines.append( line ) 

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( input_lines ) 
tokenized_input_lines = tokenizer.texts_to_sequences( input_lines ) 

length_list = list()
for token_seq in tokenized_input_lines:
    length_list.append( len( token_seq ))
max_input_length = np.array( length_list ).max()
print( 'Input max length is {}'.format( max_input_length ))

padded_input_lines = preprocessing.sequence.pad_sequences( tokenized_input_lines , maxlen=max_input_length , padding='post' )
encoder_input_data = np.array( padded_input_lines )
print( 'Encoder input data shape -> {}'.format( encoder_input_data.shape ))

input_word_dict = tokenizer.word_index
num_input_tokens = len( input_word_dict )+1
print( 'Number of Input tokens = {}'.format( num_input_tokens))


# ## Preparing input data for the Decoder

# In[8]:


output_lines = list()
for line in lines.output:
    output_lines.append( '<START> ' + line + ' <END>' )  

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( output_lines ) 
tokenized_output_lines = tokenizer.texts_to_sequences( output_lines ) 

length_list = list()
for token_seq in tokenized_output_lines:
    length_list.append( len( token_seq ))
max_output_length = np.array( length_list ).max()
print( 'Output max length is {}'.format( max_output_length ))

padded_output_lines = preprocessing.sequence.pad_sequences( tokenized_output_lines , maxlen=max_output_length, padding='post' )
decoder_input_data = np.array( padded_output_lines )
print( 'Decoder input data shape -> {}'.format( decoder_input_data.shape ))

output_word_dict = tokenizer.word_index
num_output_tokens = len( output_word_dict )+1
print( 'Number of Output tokens = {}'.format( num_output_tokens))


# ## Preparing target data for the Decoder 

# In[9]:

def decoder():
    decoder_target_data = list()
    for token_seq in tokenized_output_lines:
        decoder_target_data.append( token_seq[ 1 : ] ) 
        
    padded_output_lines = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )
    onehot_output_lines = utils.to_categorical( padded_output_lines , num_output_tokens )
    decoder_target_data = np.array( onehot_output_lines )
    print( 'Decoder target data shape -> {}'.format( decoder_target_data.shape ))


# ## Defining the Model

# In[10]:

def buildModel():
    encoder_inputs = tf.keras.layers.Input(shape=( None , ))
    encoder_embedding = tf.keras.layers.Embedding( num_input_tokens, 256 , mask_zero=True ) (encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 256 , return_state=True , recurrent_dropout=0.2 , dropout=0.2 )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
    decoder_embedding = tf.keras.layers.Embedding( num_output_tokens, 256 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( num_output_tokens , activation=tf.keras.activations.softmax ) 
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')

    model.summary()


# ## Training

# In[ ]:


def train():
    model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=124, epochs=250) 
    model.save( 'model.h5' ) 


# ## Inference models

# In[12]:


def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=(256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(256,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


# In[13]:

def inputDict():
    return input_word_dict

def outputDict():
    return output_word_dict

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( input_word_dict[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')


# In[ ]:

def saveWeights():
    enc_model , dec_model = lstm.make_inference_models()
    enc_model.save( 'enc_model.h5' ) 
    dec_model.save( 'dec_model.h5' ) 
    model.save( 'model.h5' ) 

# # In[ ]:

# import flask
# from tensorflow.keras import layers , activations , models , preprocessing , utils
# from LSTM import str_to_tokens, input_word_dict,max_output_length,max_input_length,output_word_dict
# from flask import Flask, request
# import tensorflow as tf
# import numpy as np
# from flask import Flask, request
# import tensorflow as tf
# import numpy as np
# import flask


# app = Flask(__name__)
# enc_model = tf.keras.models.load_model('enc_model.h5',custom_objects=None,compile=True)
# dec_model = tf.keras.models.load_model('dec_model.h5',custom_objects=None,compile=True)
# model = tf.keras.models.load_model('model.h5',custom_objects=None,compile=True)


# graph = tf.get_default_graph()

# # request model prediction
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'GET':
#         a = request.args['a']
#     global graph
#     with graph.as_default():
#         states_values = enc_model.predict( str_to_tokens(a ) )
#         empty_target_seq = np.zeros( ( 1 , 1 ) )
#         empty_target_seq[0, 0] = output_word_dict['start']
#         stop_condition = False
#         decoded_translation = ''
#         while not stop_condition :
#           dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
#           sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
#           sampled_word = None
#           for word , index in output_word_dict.items() :
#               if sampled_word_index == index :
#                   decoded_translation += ' {}'.format( word )
#                   sampled_word = word
          
#           if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
#               stop_condition = True
              
#           empty_target_seq = np.zeros( ( 1 , 1 ) )  
#           empty_target_seq[ 0 , 0 ] = sampled_word_index
#           states_values = [ h , c ] 
#         data = {'result': decoded_translation.replace(' end', '')}
#         return flask.jsonify(data)
# app.run(debug = False, threaded = False)
        


