from Bio import SeqIO

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import constants
from data import DNADataset
from network import CNN
from training import starting_train
# Takes in fasta file AND category for the sequence
def read_fasta(file, label): 
    # Empty list to store all sequences
    sequences = []
    # Iterate through each record in the fasta file
    for record in SeqIO.parse(file, "fasta"):
        sequences.append((str(record.seq), label))
    # Return list of sequences
    return sequences

def encode_sequence(seq):
    base_to_one_hot = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    one_hot_encoded = [base_to_one_hot[base] for base in seq]
    return [element for sublist in one_hot_encoded for element in sublist]  # Flatten the list

def main():
    accessible_sequences = read_fasta("files/accessible.fasta", 1) # 1 for accessible
    not_accessible_sequences = read_fasta("files/notaccessible.fasta", 0) # 0 for not accessible
    test_sequences = read_fasta("files/test.fasta", None) # None for test
    
    # Encode your sequences
    accessible_sequences_encoded = [(encode_sequence(seq), label) for seq, label in accessible_sequences]
    not_accessible_sequences_encoded = [(encode_sequence(seq), label) for seq, label in not_accessible_sequences]
    test_sequences_encoded = [(encode_sequence(seq), label) for seq, label in test_sequences]

    # Splitting the dataset
    train_data, val_data = train_test_split(accessible_sequences_encoded + not_accessible_sequences_encoded, test_size=0.2)
    test_data = test_sequences_encoded
    train_dataset = DNADataset(train_data)
    val_dataset = DNADataset(val_data)
    test_dataset = DNADataset(test_data)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = CNN()

    starting_train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model=model,
    hyperparameters= {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE},
    n_eval=constants.N_EVAL,
    )

if __name__ == '__main__':
    main()

