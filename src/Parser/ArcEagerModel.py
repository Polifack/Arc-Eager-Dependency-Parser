from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.layers import Input, concatenate, Dense, Embedding, Flatten, TimeDistributed, Dropout, Activation
from keras.utils import plot_model
from pickle import dump, load
from numpy import argmax

from matplotlib import pyplot as plt

class ArcEagerModel:
    '''
    Class defining the ArcEager Neural Oracle. This class is responsible for
    training and predicting the actions and relations of the ArcEager Parser.

    This class deals with the following tasks:
        - Build the dictionaries of the parser
        - Build the keras model
        - Train the keras model
        - Predict the actions and relations of the parser
    '''

    def __init__(self, seq_l, e_dim, h_dim):
        # init parser parameters
        self.seq_l = seq_l

        # as this is an arc-eager parser n_actions will always be 4
        self.a_num = 4

        # init tokenizers
        self.w_tok = None
        self.p_tok = None
        self.r_tok = None

        self.w_num = 0
        self.p_num = 0
        self.r_num = 0 

        # init keras model parameters
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.model = None
        self.hist  = None

    def build_dictionaries(self, train, dev):
        '''
        Creates and trains the tokenizers employed in the ArcEager Parser
        '''
        w_list  = []
        p_list  = []
        r_list  = []
        
        self.w_tok  = Tokenizer(filters="", oov_token="<oov>")
        self.p_tok  = Tokenizer(filters="", oov_token="<oov>")
        self.r_tok  = Tokenizer(filters="", oov_token="<oov>")

        for conll_tree in train+dev:
            w_list.append(conll_tree.get_words())
            p_list.append(conll_tree.get_postags())
            r_list.append(conll_tree.get_relations())
        
        self.w_tok.fit_on_texts(w_list)
        self.p_tok.fit_on_texts(p_list)
        self.r_tok.fit_on_texts(r_list)

        self.w_num = len(self.w_tok.word_index) + 1
        self.p_num = len(self.p_tok.word_index) + 1
        self.r_num = len(self.r_tok.word_index) + 1
    
    def build_model(self):
        ''' Build the neural tagger according to the specified config in the initialization'''
        
        word_stack_input = Input(shape=(self.seq_l,), name="ws_in")

        word_stack_emb = Embedding(
                        input_dim       = self.w_num, 
                        output_dim      = self.e_dim,
                        input_length    = self.seq_l,
                        name            = "ws_emb", 
                        mask_zero       = True)(word_stack_input)

        word_buffer_input = Input(shape=(self.seq_l,), name="wb_in")

        word_buffer_emb = Embedding(
                        input_dim       = self.w_num, 
                        output_dim      = self.e_dim,
                        input_length    = self.seq_l,
                        name            = "wb_emb", 
                        mask_zero       = True)(word_buffer_input)

        pos_stack_input = Input(shape=(self.seq_l,), name="ps_in")

        pos_stack_emb = Embedding(
                        input_dim       = self.p_num, 
                        output_dim      = self.e_dim,
                        input_length    = self.seq_l,
                        name            = "ps_emb", 
                        mask_zero       = True)(pos_stack_input)

        pos_buffer_input = Input(shape=(self.seq_l,), name="pb_in")

        pos_buffer_emb = Embedding(
                        input_dim       = self.p_num, 
                        output_dim      = self.e_dim,
                        input_length    = self.seq_l,
                        name            = "pb_emb", 
                        mask_zero       = True)(pos_buffer_input)
        
        concat = concatenate([word_stack_emb, word_buffer_emb, pos_stack_emb, pos_buffer_emb])
        flatten = Flatten()(concat)

        if self.h_dim > 0:
            hlayer = Dense(units=self.h_dim, activation = 'relu', name = 'hlayer')(flatten)
        else:
            hlayer = flatten

        action_out = Dense(units = self.a_num, activation = 'softmax', name="action")(hlayer)

        relation_out = Dense(units = self.r_num, activation = 'softmax', name="relation")(hlayer)

        self.model = Model([word_stack_input, word_buffer_input, pos_stack_input, pos_buffer_input], [action_out, relation_out])

        print("*** VOCABULARY")
        print("    words_vocab      =", self.w_num)
        print("    postags_vocab    =", self.p_num)
        print("    relations_vocab  =", self.r_num)
        print("*** PARSER MODEL")
        print("    sequence_length  =", self.seq_l)
        print("    embeddings_dim   =", self.e_dim)
        print("    dense_dim        =", self.h_dim)
    
    def compile_model(self, loss, optimizer, learning_rate, metrics):        
        # build model  
        self.build_model()

        if optimizer=='adam':
            optim  = Adam(learning_rate)
        
        elif optimizer=='sgd':
            optim = SGD(learning_rate)

        self.model.compile(optimizer=optim, loss=loss, metrics=metrics)
        print("*** COMPILATION")
        print("    optimizer        =", optimizer)
        print("    loss_fucntion    =", loss)
        print("    learning_rate    =", learning_rate)
        print("    metrics          =", metrics)

    def prepare_data(self, data, is_train=True):
        ''' Given a list of ArcEagerConfig from the arc-eager parser
            tokenizes, pads and converts to categorical the data.
        '''
        # tokenize
        w_stacks = self.w_tok.texts_to_sequences([c["sw"] for c in data])
        w_buffers = self.w_tok.texts_to_sequences([c["bw"] for c in data])
        
        p_stacks = self.p_tok.texts_to_sequences([c["sp"] for c in data])
        p_buffers = self.p_tok.texts_to_sequences([c["bp"] for c in data])

        # pad
        # nota: al hacer esto ya no necesitamos trimmear los stakcs en ArcEagerConfig
        w_stacks = pad_sequences(w_stacks, maxlen=self.seq_l, padding='pre', truncating='pre')
        w_buffers = pad_sequences(w_buffers, maxlen=self.seq_l, padding='post', truncating='post')
        
        p_stacks = pad_sequences(p_stacks, maxlen=self.seq_l, padding='pre', truncating='pre')
        p_buffers = pad_sequences(p_buffers, maxlen=self.seq_l, padding='post', truncating='post')

        # convert to categorical
        if is_train:
            a_cat = to_categorical([c["a"] for c in data], num_classes=self.a_num)
            r_cat = to_categorical(self.r_tok.texts_to_sequences([c["r"] for c in data]), num_classes=self.r_num)
        else:
            a_cat = None
            r_cat = None

        return [w_stacks, w_buffers, p_stacks, p_buffers], [a_cat, r_cat]

    def train(self, train_set, dev_set, epochs, batch_size, verbose=1):
        ''' Trains the neural tagger on the specified train_set and dev_set
            containing the arc-parser configurations.
        '''
        # prepare train data
        train_x, train_y = self.prepare_data(train_set)
        dev_x, dev_y = self.prepare_data(dev_set)

        # train
        self.hist = self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(dev_x, dev_y))

    def predict(self, data):
        ''' Predicts the action and relation for the given data.
            Returns the action and relation as a tuple.
        '''
        x, _ = self.prepare_data(data, is_train=False)
        a, r = self.model.predict(x, verbose=0)
        r = self.r_tok.sequences_to_texts([argmax(r, axis=1)])[0]
        return a, r

    def save_model(self, out_path):
        self.model.save(out_path+"/model.h5", overwrite=True)
        with open(out_path+"/model.history", 'wb') as file_pi:
            dump(self.model.history.history, file_pi)
        
        with open(out_path+"/words.tokenizer", 'wb') as file_pi:
            dump(self.w_tok, file_pi)
        
        with open(out_path+"/postags.tokenizer", 'wb') as file_pi:
            dump(self.p_tok, file_pi)
    
        with open(out_path+"/relations.tokenizer", 'wb') as file_pi:
            dump(self.r_tok, file_pi)

    def plot_history(self, save_file=False, filename="history.png"):
        h = self.hist

        fig, (fig_1, fig_2) = plt.subplots(2, figsize=(15, 15))

        fig_1.set_title('Action Accuracy')
        fig_1.plot(h['action_acc'], color='blue', label='Training')
        fig_1.plot(h['val_action_acc'], color='red', label='Validation')
        

        x_tr = len(h['action_acc'])-1
        y_tr = h['action_acc'][-1]
        text_tr = "{:.2f}".format(100*y_tr)+"%"

        fig_1.annotate(text_tr,xy=(x_tr,y_tr))

        x_val = len(h['val_action_acc'])-1
        y_val = h['val_action_acc'][-1]
        text_val = "{:.2f}".format(100*y_val)+"%"

        fig_1.annotate(text_val,xy=(x_val,y_val))

        fig_2.set_title('Relation Accuracy')
        fig_2.plot(h['relation_acc'], color='blue', label='Training')
        fig_2.plot(h['val_relation_acc'], color='red', label='Validation')
        

        x_tr = len(h['relation_acc'])-1
        y_tr = h['relation_acc'][-1]
        text_tr = "{:.2f}".format(100*y_tr)+"%"

        fig_2.annotate(text_tr,xy=(x_tr,y_tr))

        x_val = len(h['val_relation_acc'])-1
        y_val = h['val_relation_acc'][-1]
        text_val = "{:.2f}".format(100*y_val)+"%"

        fig_2.annotate(text_val,xy=(x_val,y_val))


        fig.legend(loc='lower right')
        fig.show()

        if save_file:
          plt.savefig(filename)

    def plot_model_architecture(self, path):
        plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)
    
    @staticmethod
    def from_file(model_path):
        ''' 
        Loads the model from the specified path.
        '''
        
        # retrieve model parameters
        keras_model = load_model(model_path+"/model.h5")
        seq_l       = None
        h_dim       = None
        e_dim       = None
        for layer in (keras_model.get_config()['layers']):
            if layer['class_name'] == 'Embedding':
                seq_l = layer['config']['batch_input_shape'][1]
                e_dim = layer['config']['output_dim']
            if layer['name'] == 'hlayer':
                h_dim = layer['config']['units']

        print("[*] Model loaded")
        print("    Sequence length: {}".format(seq_l))
        print("    Embedding dimension: {}".format(e_dim))
        print("    Hidden layer dimension: {}".format(h_dim))
        
        arcEagerModel = ArcEagerModel(seq_l, e_dim, h_dim)
        arcEagerModel.model = keras_model

        # load history
        with open(model_path+'/model.history', "rb") as file_pi:
            arcEagerModel.hist = load(file_pi)
        
        # load tokenizers
        with open(model_path+"/words.tokenizer", 'rb') as file_pi:
            arcEagerModel.w_tok = load(file_pi)
            arcEagerModel.w_num = len(arcEagerModel.w_tok.word_index)+1
    
        with open(model_path+"/postags.tokenizer", 'rb') as file_pi:
            arcEagerModel.p_tok = load(file_pi)
            arcEagerModel.p_num = len(arcEagerModel.p_tok.word_index)+1

        with open(model_path+"/relations.tokenizer", 'rb') as file_pi:
            arcEagerModel.r_tok = load(file_pi)
            arcEagerModel.r_num = len(arcEagerModel.r_tok.word_index)+1
        
        print("[*] Model loaded from", model_path)
        return arcEagerModel