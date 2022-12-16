# train a model with the default parameters
python main.py train --input <folder> --output <folder>

# train a model with a different buffer and stack size
python main.py train --input <folder> --output <folder> \
       --seq_l_b 2 --seq_l_s 4

# train a model with different hyperparameters
python main.py train --input <folder> --output <folder> \
       --seq_l_b 2 --seq_l_s 4 --lr 0.001 --epochs 10

# train a model with different model shapes
python main.py train --input <folder> --output <folder> \
       --seq_l_b 2 --seq_l_s 4 --lr 0.001 --epochs 10 \
       --embedding_size 100 --hidden_size 256 --drop 0.5

# train a model using pre-trained glove embeddings
# NOTE: As we are using glove-100 embeddings, we need to set the embedding size to 100
python main.py train --input <folder> --output <folder> \
       --seq_l_b 2 --seq_l_s 4 --lr 0.001 --epochs 10 \
       --embedding_size 100 --hidden_size 256 --drop 0.5 \
       --glove_path <glove_folder>