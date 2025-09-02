import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
from pathlib import Path

class LRADataLoader:
    """
    Data loader for Long Range Arena (LRA) benchmarks
    """
    
    def __init__(self, data_dir: str = "./lra_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # LRA task configurations
        self.task_configs = {
            'listops': {
                'seq_len': 2000,
                'vocab_size': 20,
                'num_classes': 10,
                'input_type': 'discrete',
                'task_type': 'classification'
            },
            'text': {
                'seq_len': 4000, 
                'vocab_size': 256,  # Character-level
                'num_classes': 2,   # Binary sentiment
                'input_type': 'discrete',
                'task_type': 'classification'
            },
            'retrieval': {
                'seq_len': 4000,
                'vocab_size': 256,
                'num_classes': 2,
                'input_type': 'discrete', 
                'task_type': 'classification'
            },
            'pathfinder32': {
                'seq_len': 1024,  # 32x32 flattened
                'vocab_size': 256,
                'num_classes': 2,
                'input_type': 'continuous',
                'task_type': 'classification'
            },
            'pathx': {
                'seq_len': 1024,  # 32x32 flattened  
                'vocab_size': 256,
                'num_classes': 2,
                'input_type': 'continuous',
                'task_type': 'classification'
            },
            'cifar10': {
                'seq_len': 1024,  # 32x32 flattened
                'vocab_size': 256, 
                'num_classes': 10,
                'input_type': 'continuous',
                'task_type': 'classification'
            }
        }
    
    def download_lra_data(self, task_name: str):
        """
        Load actual LRA data from downloaded files
        """
        if task_name == 'listops':
            return self._load_listops_data(self.data_dir)
        elif task_name == 'text':
            return self._load_text_data(self.data_dir)
        elif task_name == 'retrieval':
            return self._load_retrieval_data(self.data_dir)
        else:
            print(f"Task {task_name} not yet implemented")
            return self._create_dummy_data(task_name)
    
    def _create_dummy_data(self, task_name: str):
        """
        Fallback dummy data for unimplemented tasks
        """
        config = self.task_configs[task_name]
        
        # Generate dummy training data
        train_size = 1000
        test_size = 100
        
        if config['input_type'] == 'discrete':
            train_x = np.random.randint(0, config['vocab_size'], 
                                      (train_size, config['seq_len']))
            test_x = np.random.randint(0, config['vocab_size'], 
                                     (test_size, config['seq_len']))
        else:
            train_x = np.random.randn(train_size, config['seq_len']).astype(np.float32)
            test_x = np.random.randn(test_size, config['seq_len']).astype(np.float32)
        
        train_y = np.random.randint(0, config['num_classes'], train_size)
        test_y = np.random.randint(0, config['num_classes'], test_size)
        
        return {
            'train': {'inputs': train_x, 'targets': train_y},
            'test': {'inputs': test_x, 'targets': test_y},
            'config': config
        }
    
    def _load_listops_data(self, data_dir):
        """
        Load actual ListOps data from TSV files
        """
        train_file = data_dir / "listops-1000" / "basic_train.tsv"
        test_file = data_dir / "listops-1000" / "basic_test.tsv"
        
        def parse_tsv(filepath):
            inputs = []
            targets = []
            
            with open(filepath, 'r') as f:
                # Skip header
                next(f)
                
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            source = parts[0].strip()
                            target = int(parts[1].strip())
                            inputs.append(source)
                            targets.append(target)
            
            return inputs, targets
        
        print(f"Loading ListOps data from {data_dir}")
        train_inputs, train_targets = parse_tsv(train_file)
        test_inputs, test_targets = parse_tsv(test_file)
        
        print(f"Loaded {len(train_inputs)} training examples, {len(test_inputs)} test examples")
        
        return {
            'train': {'inputs': train_inputs, 'targets': train_targets},
            'test': {'inputs': test_inputs, 'targets': test_targets},
            'config': self.task_configs['listops']
        }

class LRADataset(Dataset):
    """
    PyTorch Dataset for LRA tasks
    """
    
    def __init__(self, inputs, targets, task_config, transform=None):
        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.task_config = task_config
        self.transform = transform
        
        # Handle different input types
        if task_config['input_type'] == 'continuous':
            self.inputs = self.inputs.float()
        else:
            self.inputs = self.inputs.long()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        # Create attention mask (all ones for now, modify as needed)
        attention_mask = torch.ones(len(x), dtype=torch.long)
        
        return {
            'input_ids': x,
            'attention_mask': attention_mask,
            'label': y
        }

class LRATextDataset(Dataset):
    """
    Specialized dataset for LRA Text task (character-level IMDb)
    """
    
    def __init__(self, texts, labels, max_length=4000, vocab_size=256):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Character-level vocabulary (ASCII)
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<UNK>'] = 1
    
    def _encode_text(self, text):
        """Convert text to character-level token ids"""
        # Convert to lowercase and encode each character
        encoded = []
        for char in text.lower()[:self.max_length]:
            encoded.append(self.char_to_id.get(char, 1))  # 1 for <UNK>
        
        # Pad to max_length
        while len(encoded) < self.max_length:
            encoded.append(0)  # 0 for <PAD>
            
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        input_ids = self._encode_text(text)
        attention_mask = (input_ids != 0).long()  # Mask padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'label': label
        }

class LRAListOpsDataset(Dataset):
    """
    Specialized dataset for LRA ListOps task with actual data format
    """
    
    def __init__(self, sequences, targets, max_length=2000):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.max_length = max_length
        
        # Build vocabulary from the actual data
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_counter = 2
        
        # Common ListOps tokens
        common_tokens = ['(', ')', '[', ']', 'MAX', 'MIN', 'MED', 'SM', 'FIRST', 'LAST', 'SUM_MOD']
        for token in common_tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_counter
                self.vocab_counter += 1
        
        # Add digits 0-9
        for i in range(10):
            token = str(i)
            if token not in self.vocab:
                self.vocab[token] = self.vocab_counter
                self.vocab_counter += 1
        
        # Build vocabulary from sequences
        for seq in sequences:
            tokens = seq.split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.vocab_counter
                    self.vocab_counter += 1
        
        print(f"Built ListOps vocabulary with {len(self.vocab)} tokens")
        print(f"Sample vocab: {list(self.vocab.items())[:20]}")
    
    def _encode_sequence(self, sequence):
        """Encode ListOps sequence to token ids"""
        tokens = sequence.split()
        encoded = []
        
        for token in tokens[:self.max_length]:
            encoded.append(self.vocab.get(token, 1))  # 1 for <UNK>
            
        # Pad to max_length
        while len(encoded) < self.max_length:
            encoded.append(0)  # 0 for <PAD>
            
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        input_ids = self._encode_sequence(sequence)
        attention_mask = (input_ids != 0).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': target
        }

def create_lra_dataloaders(task_name: str, batch_size: int = 32, data_dir: str = "/home/ubuntu/data/lra_release/lra_release"):
    """
    Create train and test dataloaders for specified LRA task
    """
    
    # Initialize data loader
    lra_loader = LRADataLoader(data_dir)
    
    # Download/load data
    data = lra_loader.download_lra_data(task_name)
    config = data['config']
    
    # Create appropriate dataset based on task
    if task_name == 'listops':
        train_dataset = LRAListOpsDataset(
            data['train']['inputs'],
            data['train']['targets'],
            max_length=config['seq_len']
        )
        test_dataset = LRAListOpsDataset(
            data['test']['inputs'],
            data['test']['targets'],
            max_length=config['seq_len']
        )
        
        # Update config with actual vocab size from dataset
        config['vocab_size'] = len(train_dataset.vocab)
        print(f"Updated vocab size to {config['vocab_size']} from ListOps data")
        
    elif task_name == 'text':
        # For actual implementation, load real IMDb character-level data
        # This is dummy data for now
        train_texts = [f"dummy text {i}" * 100 for i in range(len(data['train']['inputs']))]
        test_texts = [f"dummy text {i}" * 100 for i in range(len(data['test']['inputs']))]
        
        train_dataset = LRATextDataset(
            train_texts, 
            data['train']['targets'],
            max_length=config['seq_len']
        )
        test_dataset = LRATextDataset(
            test_texts,
            data['test']['targets'], 
            max_length=config['seq_len']
        )
        
    else:
        # General dataset for other tasks
        train_dataset = LRADataset(
            data['train']['inputs'],
            data['train']['targets'],
            config
        )
        test_dataset = LRADataset(
            data['test']['inputs'],
            data['test']['targets'],
            config
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, config

def run_lra_experiment(task_name: str, model, batch_size: int = 32, epochs: int = 10):
    """
    Run LRA benchmark experiment
    """
    
    # Create dataloaders
    train_loader, test_loader, config = create_lra_dataloaders(
        task_name=task_name,
        batch_size=batch_size
    )
    
    print(f"LRA {task_name.upper()} Task Configuration:")
    print(f"  Sequence Length: {config['seq_len']}")
    print(f"  Vocab Size: {config['vocab_size']}")
    print(f"  Num Classes: {config['num_classes']}")
    print(f"  Input Type: {config['input_type']}")
    print(f"  Task Type: {config['task_type']}")
    
    # Adapt model for this task (you'll need to modify embedding size, num_classes, etc.)
    # This assumes your model is flexible enough to handle different configs
    
    return train_loader, test_loader, config

# Usage example:
if __name__ == "__main__":
    # Test LRA data loading
    task = 'listops'  # or 'listops', 'retrieval', etc.
    
    train_loader, test_loader, config = create_lra_dataloaders(
        task_name=task,
        batch_size=32
    )
    
    # Test a batch
    for batch in train_loader:
        print("Batch shape:", batch['input_ids'].shape)
        print("Labels shape:", batch['label'].shape)
        print("Config:", config)
        break