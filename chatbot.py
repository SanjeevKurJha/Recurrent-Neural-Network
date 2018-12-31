
#Importing Libraries

import numpy as np
import re
import time
import tensorflow as tf
from collections import Counter
from tensorflow.python.layers.core import Dense


#Importing movi conversation and lines
lines = open('Data\movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('Data\movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

config_hp = tf.contrib.training.HParams(VOCAB_THRESHOLD = 20,                                 
                                        EPOCHS = 2,
                                        BATCH_SIZE = 64,
                                        RNN_SIZE = 16,
                                        NUM_LAYERS = 3,
                                        ENCODING_EMBED_SIZE = 16,
                                        DECODING_EMBED_SIZE = 16,
                                        LEARNING_RATE = 0.0001,
                                        LEARNING_RATE_DECAY = 0.9, 
                                        MIN_LEARNING_RATE = 0.0001,
                                        KEEP_PROBS = 0.5,
                                        CLIP_RATE = 4)
# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
 
# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
 
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
    

def clean_text(text):
  
    text = text.lower()
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you'r", "you are", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r" \'m", " am", text)  
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"they'r", "they are", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"this's", "this is", text)
    text = re.sub(r"'what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r'"' , "", text)
    text = re.sub(r"'" , "", text)
    text = re.sub(r"[0-9]+" , "", text)
    text = re.sub(r"<b>" , "", text)
    text = re.sub(r"<i>" , "", text)
    text = re.sub(r"<" , "", text)
    text = re.sub(r">" , "", text)  
    text = re.sub(r"[~`!@#$%^&*_=():;/?_+|,.-]","",text)
    text=text.replace("[","")
    text=text.replace("]","")
    return text
  
def clean_data():
  # Cleaning the questions
  cleaned_questions = []
  for question in questions:
      cleaned_questions.append(clean_text(question))

  # Cleaning the answers
  cleaned_answers = []
  for answer in answers:
      cleaned_answers.append(clean_text(answer))
  return cleaned_questions, cleaned_answers
                                        
 #This function is used to create vocabulary, word_to_id and id_to_word dicts from cleaned data (got from the last question).   
def create_vocab(questions, answers):

	assert len(questions) == len(answers)
	vocab = []
	for i in range(len(questions)):
		words = questions[i].split()
		for word in words:
			vocab.append(word)

		words = answers[i].split()
		for word in words:
			vocab.append(word)

	vocab = Counter(vocab)
	tokens = []
	for key in vocab.keys():
		if vocab[key] >= config_hp.Vocab_Threshold:
			tokens.append(key)

	tokens = ['<PAD>', '<SOS>', '<UNK>', '<EOS>'] + tokens

	word_to_id = {word:i for i, word in enumerate(tokens)}
	id_to_word = {i:word for i, word in enumerate(tokens)}

	return tokens, word_to_id, id_to_word

#Using word_to_id dictionery to map each word in the sample to it's own int representation
def encoder_word_to_id(data, word_to_id, targets=False):

	encoded_data = []

	for i in range(len(data)):

		encoded_line = []
		words = data[i].split()
		for word in words:

			if word not in word_to_id.keys():
				encoded_line.append(word_to_id['<UNK>'])
			else:
				encoded_line.append(word_to_id[word])

		if targets:
			encoded_line.append(word_to_id['<EOS>'])

		encoded_data.append(encoded_line)


	return np.array(encoded_data)


############ Graph Functions ##################
#graph_inputs function is used to define all tensorflow graph placeholders


def graph_inputs():
  
    #Inputs placeholder will be fed with question sentence data, and its shape is [None, None]
    #The first None means the batch size, and the batch size is unknown since user can set it
    #The second None means the lengths of sentences.
    en_inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    
    #en_targets placeholder is similar to inputs placeholder except that it will be fed with answer sentence data.
    en_targets = tf.placeholder(tf.int32, [None, None], name='targets')
    
    #keep_probs - probabilities used in DropoutWrapper in dropout layer(generally we are using it for generalization of the model)
    keep_probs = tf.placeholder(tf.float32, name='dropout_rate')
    
    #encoder_seq_len - vector placeholder represents the lengths of each sentences, so the shape is None
    encoder_seq_len = tf.placeholder(tf.int32, (None, ), name='encoder_seq_len')
    
    #decoder_seq_len - vector which is used to define lengths of each sample in the targets to the model
    decoder_seq_len = tf.placeholder(tf.int32, (None, ), name='decoder_seq_len')
    
    #max_seq_len - gets the maximum value out of lengths of all the target sentences(sequences)
    max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')
    
    return en_inputs, en_targets, keep_probs, encoder_seq_len, decoder_seq_len, max_seq_len


def encoder_rnn_layer(inputs, rnn_size, number_of_layers, encoder_seq_len, keep_probs, encoder_embed_size, encoder_vocab_size):
                
    #rnn_size: int value, The number of units in the LSTM cell.
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
                
    #An rnn_cell, a projection to output_size is added to it.
    #keep_probs: unit Tensor or float between 0 and 1, if it is constant and 1, no input dropout will be added
    rnn_cell= tf.contrib.rnn.DropoutWrapper(lstm, keep_probs)
                
    #encoder_cell: Composed sequentially of a number of rnn_cell.
    encoder_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * number_of_layers)
                
    #encoder_embedding: Each word in a sentence will be represented with the number of features specified as encoder_embed_size
    encoder_embedings = tf.contrib.layers.embed_sequence(inputs, encoder_vocab_size, encoder_embed_size) 
                
    #Put embeding layer and rnn stacked layer all togather 
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, encoder_embedings,encoder_seq_len,dtype=tf.float32)

    return encoder_outputs, encoder_states

def preprocessing_target(targets, word_to_id, batch_size):
	  #This line is used to REMOVE last member of each sample in the decoder_inputs batch
    #Stride, we can think like we are splitting it with multiple stride window with some size of window   
    endings = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])  
    #tf.fill we can see we are creating tensor field with scaler value                        
    #returning line and in this line we concat '<SOS>' tag at the beginning of each sample in the batch
    decoder_inputs= tf.concat([tf.fill([batch_size, 1], word_to_id['<SOS>']), endings], 1) 
                
    return decoder_inputs


def decoder_rnn_layer(decoder_inputs, encoder_states, decoder_cell, decoder_embed_size, vocab_size, decoder_seq_len, max_seq_len, word_to_id, batch_size):

    #Defining embedding layer for the Decoder, This is used to convert encode training target texts to list of ids.
    embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
    embeding_matrix = tf.nn.embedding_lookup(embed_layer, decoder_inputs) 
               
    # Creating Dense (Fully Connected) layer at the end of the Decoder, a neural network operates on dense vectors of some size,
    # often 256, 512 or 1024 floats (let's say 256 for here).But at the end it needs to predict a word from the vocabulary which is often much larger,    
    # e.g., 40000 words. Output projection is the final linear layer that converts (projects) from the internal representation to the larger one.
    # So, for example, it can consist of a 256 x 20000 parameter matrix and a 20000 parameter for the bias vector.
    projection_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    
    with tf.variable_scope('decod'):
        #Training helper used only to read inputs in the embeded stage,As the name indicates, this is only a helper instance.
        # This instance should be delivered to the BasicDecoder, which is the actual process of building the decoder model.
        train_helper = tf.contrib.seq2seq.TrainingHelper(embeding_matrix, decoder_seq_len)
        
        #Defining decoder - You can change with BeamSearchDecoder, just beam size
        #BasicDecoder builds the decoder model. It means it connects the RNN layer(s) on the decoder side and the input prepared by TrainingHelper
        train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_states,projection_layer)
        
        #dynamic_decode unrolls the decoder model so that actual prediction can be retrieved by BasicDecoder for each time steps.
        train_dec_output, train_dec_state, train_dec_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(train_decoder, 
                                                                                                          impute_finished=True, 
                                                                                                          maximum_iterations=max_seq_len)
               
    #we are using reuse=True option in this scope because we want to get same params learned in the previouse 'decoder' scope    
    with tf.variable_scope('decod', reuse=True): 
        #getting vector of the '<SOS>' tags in the int representation
        starting_id_vec = tf.tile(tf.constant([word_to_id['<SOS>']], dtype=tf.int32), [batch_size], name='starting_id_vec')
                
        #GreedyEmbeddingHelper dynamically takes the output of the current step and give it to the next time step’s input. 
        #In order to embed the each input result dynamically, embedding parameter(just bunch of weight values) should be provided
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer, starting_id_vec, word_to_id['<EOS>'])

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,inference_helper, encoder_states, projection_layer)
        
        inference_dec_output, inference_dec_state, inference_dec_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(inference_decoder, 
                                                                                                                      impute_finished=True, 
                                                                                                                      maximum_iterations=max_seq_len)
        
    return train_dec_output, inference_dec_output


def attention_model(rnn_size, keep_probs, encoder_outputs, encoder_states, encoder_seq_len, batch_size):
    #rnn_size: int value, The number of units in the LSTM cell.
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    #An rnn_cell, a projection to output_size is added to it.
    #keep_probs: unit Tensor or float between 0 and 1, if it is constant and 1, no input dropout will be added
    decoder_cell=tf.contrib.rnn.DropoutWrapper(lstm, keep_probs)
    
    #using helper function from seq2seq sub_lib for Bahdanau attention
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_outputs, encoder_seq_len)    
    
    #finishin attention with the attention holder - Attention Wrapper
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, rnn_size/2)
    
    #Here we are usingg zero_state of the LSTM (in this case) decoder cell, and feed the value of the last encoder_state to it
    attention_zero = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    enc_state_new = attention_zero.clone(cell_state=encoder_states[-1])
    
    return dec_cell, enc_state_new


def opt_loss(outputs, targets, dec_seq_len, max_seq_len, learning_rate, clip_rate):
   #out put is a predicted value 
    logits = tf.identity(outputs.rnn_output)
    
    mask_weigts = tf.sequence_mask(dec_seq_len, max_seq_len, dtype=tf.float32)
    
    with tf.variable_scope('opt_loss'):
        #using sequence_loss to optimize the seq2seq model
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, mask_weigts)                                                       
        #Define optimizer
        opt = tf.train.AdamOptimizer(learning_rate)

        #Next 3 lines used to clip gradients {Prevent gradient explosion problem}
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_rate)
        traiend_opt = opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()))
        
    return loss, traiend_opt

class Chatbot(object):
    
    def __init__(self,rnn_size,enc_embed_size,dec_embed_size, learning_rate, batch_size, 
                 number_of_layers, vocab_size, word_to_id, clip_rate):
        
        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_probs, self.encoder_seq_len, self.decoder_seq_len, max_seq_len = graph_inputs()
        
        
        enc_outputs, enc_states = encoder_rnn_layer(self.inputs, rnn_size,number_of_layers,self.encoder_seq_len,
                                                    self.keep_probs,enc_embed_size,vocab_size)
        
        dec_inputs = preprocessing_target(self.targets, word_to_id,batch_size)
        
        decoder_cell, encoder_states_new = attention_model(rnn_size, self.keep_probs, enc_outputs,enc_states, 
                                                          self.encoder_seq_len,batch_size)
        
        train_outputs, inference_output = decoder_rnn_layer(dec_inputs, encoder_states_new,decoder_cell,dec_embed_size,
                                                              vocab_size, self.decoder_seq_len,max_seq_len,word_to_id,batch_size)
          
        self.predictions  = tf.identity(inference_output.sample_id, name='preds')
        
        self.loss, self.opt = opt_loss(train_outputs, self.targets,self.decoder_seq_len, max_seq_len,learning_rate,clip_rate)
        
 ######### Accuracy function##############
def accuracy(target, logits):
    max_seq_length = max(target.shape[1], logits.shape[1])
    if max_seq_length - target.shape[1]:
        target = np.pad(target, [(0,0),(0,max_seq_length - target.shape[1])],'constant')
    if max_seq_length - logits.shape[1]:
        logits = np.pad(logits,[(0,0),(0,max_seq_length - logits.shape[1])],'constant')

    return np.mean(np.equal(target, logits))
  
cleaned_questions, cleaned_answers = clean_data()

vocab, word_to_id, id_to_word = create_vocab(cleaned_questions, cleaned_answers)

encoded_questions =encoder_word_to_id(cleaned_questions, word_to_id)

encoded_answers = encoder_word_to_id(cleaned_answers, word_to_id, True)

chatbot_class = Chatbot(config_hp.Rnn_Size,
                config_hp.Encod_Embed_Size, 
                config_hp.Decode_Embed_Size,
                config_hp.Learning_Rate, 
                config_hp.Batch_Size,                  
                config_hp.Num_Layers,
                len(vocab), 
                word_to_id, 
                config_hp.Clip_Rate) 

# Padding the sequences with the <PAD> token
#If the sentence is shorter then wanted length, pad it to that length
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word_to_id['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
  
# Splitting the data into batches of questions and answers
def split_q_a_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, word_to_id))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, word_to_id))
        
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(cleaned_questions) * 0.15)
training_questions = encoded_questions[training_validation_split:]
training_answers = encoded_answers[training_validation_split:]
validation_questions = encoded_questions[:training_validation_split]
validation_answers = encoded_answers[:training_validation_split]

batch_index_check = ((len(training_questions)) // config_hp.Batch_Size // 2) - 1
validation_loss = []
early_stopping = 1000	

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)
for i in range(config_hp.Epochs):    
    train_accuracy = []
    train_loss = []
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_q_a_batches(training_questions, training_answers, config_hp.Batch_Size)):
        starting_time = time.time()
        feed_dict = {chatbot_class.inputs:padded_questions_in_batch, 
                     chatbot_class.targets:padded_answers_in_batch, 
                     chatbot_class.keep_probs:config_hp.Keep_Prob, 
                     chatbot_class.decoder_seq_len:[len(padded_answers_in_batch[0])]*config_hp.Batch_Size,
                     chatbot_class.encoder_seq_len:[len(padded_answers_in_batch[0])]*config_hp.Batch_Size}
        
        cost, _, preds = session.run([chatbot_class.loss, chatbot_class.opt, chatbot_class.predictions], feed_dict=feed_dict)
            
        train_accuracy.append(accuracy(np.array(padded_answers_in_batch), np.array(preds)))      
        train_loss.append(cost)
        ending_time = time.time()
        batch_time = ending_time - starting_time        
        if batch_index % batch_index_check == 0:
          print("EPOCH: {}/{}".format(i, config_hp.Epochs), 
                " | Epoch train loss: {}".format(np.mean(train_loss)), 
                " | Epoch train accuracy: {}".format(np.mean(train_accuracy)),
                " | Batch train time: {}".format(batch_time))
        if batch_index % batch_index_check == 0 and batch_index > 0:
            total_validation_loss = []
            val_accuracy=[]
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_q_a_batches(validation_questions, validation_answers, config_hp.Batch_Size)):
              feed_dict = {chatbot_class.inputs:padded_questions_in_batch, 
                           chatbot_class.targets:padded_answers_in_batch, 
                           chatbot_class.keep_probs:1, 
                           chatbot_class.decoder_seq_len:[len(padded_answers_in_batch[0])]*config_hp.Batch_Size,
                           chatbot_class.encoder_seq_len:[len(padded_answers_in_batch[0])]*config_hp.Batch_Size}
              
              batch_validation_loss,preds = session.run([chatbot_class.loss,chatbot_class.predictions], feed_dict=feed_dict)
              
              total_validation_loss.append(batch_validation_loss)
              val_accuracy.append(accuracy(np.array(padded_answers_in_batch), np.array(preds)))
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss = np.mean(total_validation_loss)
            print('Validation Loss: {:>6.3f}, Batch Validation Time: {:d} seconds, Validation Accuracy:{}'.format(average_validation_loss, int(batch_time), np.mean(val_accuracy)))
            config_hp.Learning_Rate *= config_hp.Learning_Rate
            if config_hp.Learning_Rate < config_hp.Min_Learning_Rate:
                learning_rate = config_hp.Min_Learning_Rate
            validation_loss.append(average_validation_loss)
            if average_validation_loss <= min(validation_loss):
                print('I can speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                #saver.save(session, "checkpoint/chatbot_{}.ckpt".format(i))	
            else:
                print("Please train me more, So i will speak better then now .")
                early_stopping_check += 1
                if early_stopping_check == early_stopping:
                    break
    if early_stopping_check == early_stopping:
        print("My apologies, This is the best I can do.")
        break
print("All The best!!")
