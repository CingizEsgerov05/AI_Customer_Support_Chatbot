# train.py - Professional Training Pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pickle
from backend import ChatbotDataset, BERTChatbot, DEVICE
from sklearn.model_selection import train_test_split
import numpy as np

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model():
    print("=" * 60)
    print("🚀 AI Chatbot Təlim Sistemi")
    print("=" * 60)
    
    # Dataset yüklə və artır
    print("\n📊 Məlumatlar hazırlanır...")
    dataset_obj = ChatbotDataset()
    raw_data = dataset_obj.augment_data()  # Data augmentation
    print(f"✓ Cəmi nümunə sayı: {len(raw_data)}")
    
    # Label mapping
    unique_intents = list(set([item['intent'] for item in raw_data]))
    intent_to_label = {intent: i for i, intent in enumerate(unique_intents)}
    label_to_intent = {i: intent for i, intent in enumerate(unique_intents)}
    print(f"✓ Kateqoriya sayı: {len(unique_intents)}")
    
    # Mətnləri və labelləri ayır
    texts = [item['text'] for item in raw_data]
    labels = [intent_to_label[item['intent']] for item in raw_data]
    
    # Train-Validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"✓ Təlim: {len(train_texts)}, Test: {len(val_texts)}")
    
    # Tokenizer
    print("\n🔤 Tokenizasiya...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)
    
    # Dataset və DataLoader
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Model
    print("\n🧠 Model qurulur...")
    model = BERTChatbot(num_intents=len(unique_intents), dropout=0.3)
    model.to(DEVICE)
    print(f"✓ Device: {DEVICE}")
    
    # Optimizer və Scheduler
    epochs = 20
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n🏋️ Təlim başlayır ({epochs} epoxa)...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Progress
        print(f"Epoxa {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Dəqiqlik: {train_acc:.2f}%")
        print(f"  Valid - Loss: {val_loss:.4f}, Dəqiqlik: {val_acc:.2f}%")
        
        # Early stopping və best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_chatbot_model.pth')
            print("  ✓ Ən yaxşı model yadda saxlanıldı!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping (patience={patience})")
                break
        
        print("-" * 60)
    
    # Metadata saxla
    print("\n💾 Metadata saxlanılır...")
    metadata = {
        'label_to_intent': label_to_intent,
        'dataset_obj': dataset_obj,
        'num_intents': len(unique_intents),
        'training_samples': len(train_texts),
        'validation_accuracy': val_acc
    }
    
    with open('chatbot_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "=" * 60)
    print("✅ TƏLİM UĞURLA BAŞA ÇATDI!")
    print("=" * 60)
    print(f"📈 Son validasiya dəqiqliyi: {val_acc:.2f}%")
    print(f"📁 Model: best_chatbot_model.pth")
    print(f"📁 Metadata: chatbot_metadata.pkl")
    print("\n🚀 İndi chatbot-u işə salın:")
    print("   python3 -m streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    train_model()