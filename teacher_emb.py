import numpy as np
import torch
import argparse
from sklearn.decomposition import PCA
from simcse import SimCSE
from datasets import load_dataset
from get_data import get_data

def get_sentences(file_path):
    dataset = load_dataset('csv', data_files=file_path, cache_dir="./data/", delimiter="\t" if file_path.endswith('tsv') else ",")
    sentences = set()
    ''' 
    for row in dataset['train']:
       sentences.add(row['sent0']) 
       sentences.add(row['sent1']) 
       sentences.add(row['hard_neg']) 
    '''
    for row in dataset['train']:
       sentences.add(row['sentence1']) 
       sentences.add(row['sentence2']) 
    sentences = sorted(list(sentences))
    return sentences

def get_teacher_emb(teacher='simcse', final_dim=128, batch_size=256, file_path='data/nli_for_simcse.csv', save_dir='./embs/'):
    train_sentences, valid_sentences = get_data(aug_data=True)
    # train_sentences, valid_sentences = get_msmarco_data()
    # train_sentences, valid_sentences = get_nli_data()
    # train_sentences, valid_sentences = get_msmarco_data(train_data_dir='datasets/msmarco_queries.txt',valid_data_dir='datasets/msmarco_queries_valid.txt')
    # train_sentences = train_sentences[:10000]
    # You can also get other state-of-the-art sentence embeddings by changing the teacher model
    # train_sentences = get_sentences(file_path)
    if teacher == 'simcse':
        model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
        teacher_dim = 1024
        train_embeddings = model.encode(train_sentences, batch_size=batch_size)
    elif teacher == 'st':
        # model = SentenceTransformer('stsb-roberta-base-v2')
        # teacher_dim = 1024
        # model = SentenceTransformer('stsb-mpnet-base-v2')
        # teacher_dim = 768
        model = SentenceTransformer('nli-mpnet-base-v2')
        teacher_dim = 768
        train_embeddings = torch.tensor(model.encode(train_sentences, batch_size=batch_size))
    elif teacher == 'coCondenser':
        model = SimCSE('Luyu/co-condenser-marco-retriever',pooler='cls_before_pooler')
        teacher_dim = 768
        train_embeddings = model.encode(train_sentences, batch_size=batch_size)
    else:
        raise ValueError("No Teacher Model available")

    print(train_embeddings.shape)
    train_embeddings = train_embeddings.to(torch.float32)
    train_file_path = save_dir + teacher + '-train-F' + str(train_embeddings.size(-1)) + '.pt'
    torch.save(train_embeddings.double(), train_file_path)

    pca = PCA(n_components=final_dim)
    pca.fit(train_embeddings[0:40000])
    pca_comp = np.asarray(pca.components_)

    dense = torch.nn.Linear(teacher_dim, final_dim, bias=False)
    dense.weight = torch.nn.Parameter(torch.tensor(pca_comp).to(torch.float32))
    train_embeddings = train_embeddings.to(torch.float32)
    with torch.no_grad(): 
        train_embeddings = dense(train_embeddings)

    train_file_path = save_dir + teacher + '-train-F' + str(final_dim) + '.pt'
    # torch.save(dense(train_embeddings.double()), train_file_path)
    # torch.save(dense(valid_embeddings.double()), valid_file_path)

    torch.save(train_embeddings.double(), train_file_path)


    print('Finish teacher embedding, save to', train_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for getting teacher's embeddings")
    parser.add_argument("--teacher", type=str, default='simcse', choices=['simcse', 'st', 'coCondenser'], help='teacher model')
    parser.add_argument("--final-dim", type=int, default=128, help="final dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--save-dir", type=str, default='./embs/', help="save path")
    parser.add_argument("--file-path", type=str, default='data/nli_for_simcse.csv')
    args = parser.parse_args()
    get_teacher_emb(teacher=args.teacher, final_dim=args.final_dim, batch_size=args.batch_size, file_path=args.file_path, save_dir=args.save_dir)
