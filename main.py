from src.ArcEagerModel import ArcEagerModel
from src.Parser.ArcEager import ArcEagerParser
from src.ConllTree.ConllTree import ConllTree

files_path = "data/UD_English-EWT/en_ewt-ud-"

seq_l = 2

test  = ConllTree.read_conllu_file(files_path+"test.conllu")
dev   = ConllTree.read_conllu_file(files_path+"dev.conllu")
train = ConllTree.read_conllu_file(files_path+"train.conllu")

# crete the arc-eager parser
parser = ArcEagerParser(seq_l)

# create the arc-eager neural oracle
model = ArcEagerModel(seq_l, train, dev, 100, 100)

# generate the training and dev config
test_config = parser.train_on_list(test)
dev_config  = parser.train_on_list(dev)

# compile the model
model.compile_model(loss="categorical_crossentropy", optimizer="adam", learning_rate=0.001, metrics=["acc"])

# train the model
history = model.train(test_config, dev_config, 2, 32, 1)