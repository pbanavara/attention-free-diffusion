import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import time
import os

def run_experiment(dataset_name='imdb', 
                   batch_size=16, epochs=10, max_length=4096):
    """
    Run a complete training experiment with specified parameters
    """
    
    # Fixed hyperparameters
    EMBED_DIM = 256
    NUM_ITERS = 4
    ALPHA = 0.5
    LR = 5e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting experiment: {dataset_name}, batch_size={batch_size}, epochs={epochs}")
    print(f"Using device: {DEVICE}")
    
    # Dataset loading
    if dataset_name == 'imdb':
        dataset = load_dataset('imdb')
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']  # Already 0/1 for IMDB
        test_texts = dataset['test']['text']  
        test_labels = dataset['test']['label']
        num_classes = 2
        
    elif dataset_name == 'ag_news':
        dataset = load_dataset('ag_news')
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(dataset['train']['label'])
        test_labels = label_encoder.transform(dataset['test']['label'])
        train_texts = dataset['train']['text']
        test_texts = dataset['test']['text']
        num_classes = 4
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Custom Dataset Class
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Create datasets and loaders
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model (your improved diffusion model)
    class ImprovedDiffusionAttentionFreeModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_iters=4, alpha=0.5, num_classes=2):
            super().__init__()
            
            self.embed_dim = embed_dim
            self.num_iters = num_iters
            self.alpha = alpha
            self.noise_std = 0.05
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.neighbor_proj = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(3)
            ])
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.update_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes)
            )
            self._init_weights()
        
        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)
        
        def forward(self, input_ids, attention_mask):
            h = self.embedding(input_ids)
            
            if self.training:
                noise = torch.randn_like(h, dtype=h.dtype, device=h.device) * self.noise_std
                h = h + noise
            
            for iteration in range(self.num_iters):
                h_left = torch.cat([h[:, -1:, :], h[:, :-1, :]], dim=1)
                h_right = torch.cat([h[:, 1:, :], h[:, :1, :]], dim=1)
                
                h_left_proj = self.neighbor_proj[0](h_left)
                h_right_proj = self.neighbor_proj[1](h_right)
                h_self_proj = self.neighbor_proj[2](h)
                
                neighbor_sum = h_left_proj + h_right_proj + h_self_proj
                h_update = self.update_mlp(neighbor_sum)
                h_new = self.alpha * h + (1 - self.alpha) * h_update
                h = self.layer_norm(h_new)
            
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                h_masked = h * mask_expanded
                mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-8)
                pooled = h_masked.sum(dim=1) / mask_sum
            else:
                pooled = h.mean(dim=1)
            
            logits = self.classifier(pooled)
            return logits
    
    # Initialize model
    model = ImprovedDiffusionAttentionFreeModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_iters=NUM_ITERS,
        alpha=ALPHA,
        num_classes=num_classes
    ).to(DEVICE)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, eps=1e-8)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    # Memory monitoring
    def print_memory_stats():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    print_memory_stats()
    
    # Training loop
    results = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'epoch_times': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 1000 == 0:
                current_acc = 100. * correct / total_samples
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {current_acc:.2f}%")
        
        train_acc = correct / total_samples
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE) 
                labels = batch['label'].to(DEVICE)
                
                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        epoch_time = time.time() - start_time
        
        # Store results
        results['train_losses'].append(avg_train_loss)
        results['train_accs'].append(train_acc)
        results['test_losses'].append(avg_test_loss)
        results['test_accs'].append(test_acc)
        results['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Test  - Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print_memory_stats()
        print("-" * 50)
    
    return results, model

# Batch size experiment runner
def run_batch_size_experiments(dataset_name: str='imdb', 
                               batch_sizes : list[int]=[16, 32, 64], epochs=10):
    """
    Run experiments with different batch sizes
    """
    all_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT WITH BATCH_SIZE={batch_size}")
        print(f"{'='*60}\n")
        
        try:
            results, model = run_experiment(
                dataset_name=dataset_name,
                batch_size=batch_size,
                epochs=epochs,
                max_length=4096
            )
            
            all_results[batch_size] = results
            
            # Save model checkpoint
            checkpoint_path = f"diffusion_{dataset_name}_batch{batch_size}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'batch_size': batch_size,
                'final_test_acc': results['test_accs'][-1]
            }, checkpoint_path)
            
            print(f"Saved checkpoint: {checkpoint_path}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM Error with batch_size={batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
        
        # Clear memory between experiments
        torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*60)
    print("BATCH SIZE COMPARISON")
    print("="*60)
    
    for batch_size, results in all_results.items():
        final_acc = results['test_accs'][-1]
        avg_epoch_time = sum(results['epoch_times']) / len(results['epoch_times'])
        print(f"Batch Size {batch_size}: Final Acc = {final_acc:.4f}, Avg Epoch Time = {avg_epoch_time:.1f}s")
    
    return all_results

# Usage examples:
if __name__ == "__main__":
    # Single experiment
    # results, model = run_experiment('imdb', batch_size=32, epochs=10)
    
    # Multiple batch size experiments
    all_results = run_batch_size_experiments('imdb', batch_sizes=[16, 32, 64], epochs=10)