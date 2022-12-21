from src.Parser.ArcEagerModel import ArcEagerModel
from src.Parser.ArcEager import ArcEagerParser
from src.ConllTree.ConllTree import ConllTree

in_folder = "./data/UD_English-ParTUT"

# load the data
train  = ConllTree.read_conllu_file(in_folder + "/train.conllu", filter_projective=True)
dev  = ConllTree.read_conllu_file(in_folder + "/dev.conllu", filter_projective=True)

# create parser and model
#arcEagerModel = ArcEagerModel(args.seq_l_s, args.seq_l_b, args.embedding_size, args.hidden_size)


seq_l_b = 2
seq_l_s = 2
parser = ArcEagerParser(seq_l_s, seq_l_b)

for t in train[0:1]:
    c = parser.get_parser_config(t)
    if not parser.assert_sentence(t):
        print("Sentence not valid")
        print(t)
        break
