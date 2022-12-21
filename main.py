import argparse
from src.Parser.ArcEagerModel import ArcEagerModel
from src.Parser.ArcEager import ArcEagerParser
from src.ConllTree.ConllTree import ConllTree

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Arc Eager Parser')

    parser.add_argument('mode', metavar='model', type=str, choices=['train', 'eval', 'plot'], help='train or predict')
    parser.add_argument('--input', type=str, help='folder path to input data (train, dev, test)')
    parser.add_argument('--output', type=str, help='folder path to save the model or the predictions')
    
    # training arguments
    parser.add_argument('--seq_l_b', type=int, default=2, help='sequence length for buffer')
    parser.add_argument('--seq_l_s', type=int, default=2, help='sequence length for stack')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop', 'adagrad'], help='optimizer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'], help='activation function of hidden layer')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding size')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--glove_path', type=str, default=None, help='glove path')

    # evaluation arguments
    parser.add_argument('--model', type=str, help='saved model path')
    parser.add_argument('--postprocess', action='store_true', default=True, help='postprocess the predictions')
    parser.add_argument('--multi-root', action='store_false', default=True, help='allow multi-root trees')
    parser.add_argument('--n_samples', type=int, default=-1, help='number of samples to predict')

    args = parser.parse_args()

    if args.mode == 'train':
        print("[*] Train mode")
        # load the data
        train  = ConllTree.read_conllu_file(args.input + "/train.conllu", filter_projective=False)
        dev  = ConllTree.read_conllu_file(args.input + "/dev.conllu", filter_projective=False)
        
        # create parser and model
        seq_l_lc=2
        seq_l_rc=2
        arcEagerModel = ArcEagerModel(args.seq_l_s, args.seq_l_b, seq_l_lc, seq_l_rc, args.embedding_size, args.hidden_size, args.activation)
        arcEagerParser = ArcEagerParser(args.seq_l_s, args.seq_l_b, seq_l_lc, seq_l_rc)
        
        print("[*] Building the model...")
        # build model tokenizers
        arcEagerModel.build_dictionaries(train, dev)

        # generate the training and dev config
        train_config = arcEagerParser.get_parser_config_list(train)
        dev_config  = arcEagerParser.get_parser_config_list(dev)

        print("[*] Compiling the model and starting train...")
        arcEagerModel.compile_model(loss="categorical_crossentropy", optimizer=args.optimizer, learning_rate=args.lr, metrics=["acc"])
        history = arcEagerModel.train(train_config, dev_config, args.epochs, args.batch_size, 1)
        
        arcEagerModel.save_model(args.output + "/model")
    
    elif args.mode == 'plot':
        print("[*] Plot mode")
        arcEagerModel = ArcEagerModel.from_file(args.model)
        arcEagerModel.plot_history(True, args.output + "/model.png")
        arcEagerModel.plot_model_architecture(args.output + "/model_architecture.png")

    elif args.mode == 'eval':
        print("[*] Evaluation mode")
        print("    Postprocess: ", args.postprocess)
        print("    Multi-root: ", args.multi_root)
        
        # load the data cleaning multi-expression and empty tokens and NOT filtering projective trees
        if args.n_samples > 0:
            test_gold = ConllTree.read_conllu_file(args.input + "/test.conllu", filter_projective=False, dummy_root=False)[:args.n_samples]
            test  = ConllTree.read_conllu_file(args.input + "/test.conllu", filter_projective=False)[:args.n_samples]
        else:
            test_gold = ConllTree.read_conllu_file(args.input + "/test.conllu", filter_projective=False, dummy_root=False)
            test  = ConllTree.read_conllu_file(args.input + "/test.conllu", filter_projective=False)

        test_w = [t.get_words() for t in test]
        test_p = [t.get_postags() for t in test]

        arcEagerModel, seq_l_s, seq_l_b = ArcEagerModel.from_file(args.model)
        arcEagerParser = ArcEagerParser(seq_l_s, seq_l_b)

        print("[*] Predicting the test set...")
        test_pred = arcEagerParser.predict_dependency_tree_list(test_w, test_p, arcEagerModel)

        if args.postprocess:
            print("[*] Postprocessing the test set...")
            for tree in test_pred:
                tree.postprocess_tree(root_uniqueness=args.multi_root)
        
        print("[*] Writing the test set...")
        ConllTree.write_conllu_file(args.output + "/test_clean.conllu", test_gold)
        ConllTree.write_conllu_file(args.output + "/test_pred.conllu", test_pred)
