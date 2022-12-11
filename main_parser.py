import argparse
from src.Parser.ArcEagerModel import ArcEagerModel
from src.Parser.ArcEager import ArcEagerParser
from src.ConllTree.ConllTree import ConllTree

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Arc Eager Parser')

    parser.add_argument('mode', metavar='model', type=str, choices=['train', 'eval'], help='train or predict')
    parser.add_argument('--model', type=str, help='saved model path')
    parser.add_argument('--input', type=str, help='folder path to input data (train, dev, test)')
    parser.add_argument('--output', type=str, help='folder path to save the model or the predictions')
    parser.add_argument('--seq_l', type=int, default=2, help='sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='optimizer')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding size')
    parser.add_argument('--postprocess', action='store_true', default=True, help='postprocess the predictions')

    args = parser.parse_args()

    if args.mode == 'train':
        print("[*] Train mode")
        # load the data
        train  = ConllTree.read_conllu_file(args.input + "/train.conllu", filter_projective=False)
        dev  = ConllTree.read_conllu_file(args.input + "/dev.conllu", filter_projective=False)
        
        # create parser and model
        arcEagerModel = ArcEagerModel(args.seq_l, args.embedding_size, args.hidden_size)
        arcEagerParser = ArcEagerParser(args.seq_l)
        
        print("[*] Building the model...")
        # build model tokenizers
        arcEagerModel.build_dictionaries(train, dev)

        # generate the training and dev config
        train_config = arcEagerParser.get_parser_config_list(train)
        dev_config  = arcEagerParser.get_parser_config_list(dev)

        print("[*] Compiling the model and starting train...")
        arcEagerModel.compile_model(loss="categorical_crossentropy", optimizer=args.optimizer, learning_rate=args.lr, metrics=["acc"])
        history = arcEagerModel.train(train_config, dev_config, args.epochs, args.batch_size, 1)
        
        arcEagerModel.save(args.output + "/model")
    
    elif args.mode == 'eval':
        print("[*] Evaluation mode")
        test  = ConllTree.read_conllu_file(args.input + "/test.conllu", filter_projective=False)
        test_w = [t.get_words() for t in test]
        test_p = [t.get_postags() for t in test]

        arcEagerModel = ArcEagerModel.from_file(args.model)
        arcEagerParser = ArcEagerParser(args.seq_l)

        print("[*] Predicting the test set...")
        test_pred = arcEagerParser.predict_dependency_tree_list(test_w, test_p, arcEagerModel)

        if args.postprocess:
            print("[*] Postprocessing the test set...")
            for tree in test_pred:
                tree.postprocess_tree()
        
        print("[*] Writing the test set...")
        ConllTree.write_conllu_file(args.output + "/test_pred.conllu", test_pred)
