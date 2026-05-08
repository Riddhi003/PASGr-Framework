import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import matplotlib.pyplot as plt
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from sklearn.model_selection import train_test_split
from torch.distributions.laplace import Laplace
import random

# Set global seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. PASGREC ARCHITECTURE ---
class PASGRec(nn.Module):
    def __init__(self, num_nodes, num_users, embedding_dim=64):
        super().__init__()
        # Using GNN backbone as per your research
        self.num_users = num_users
        self.gnn = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=3)
    
    def forward(self, edge_index, inject_noise=True):
        # Representation Learning: GNN backbone
        emb = self.gnn.get_embedding(edge_index)
        if inject_noise and self.training:
            # Privacy Preservation: Inject Laplacian noise into user embeddings
            dist = Laplace(torch.tensor([0.0]), torch.tensor([0.005]))
            noise = dist.sample(emb[:self.num_users].shape).to(emb.device).squeeze()
            user_emb_noisy = emb[:self.num_users] + noise
            emb = torch.cat([user_emb_noisy, emb[self.num_users:]], dim=0)
        return emb

    def bpr_loss(self, embeddings, edge_index, num_users, num_nodes, weights=None):
        # edge_index[0] = users, edge_index[1] = items
        users = edge_index[0]
        pos_items = edge_index[1]
        
        # Sample negative items (must be in the item index range)
        neg_items = torch.randint(num_users, num_nodes, (users.size(0),))
        
        user_emb = embeddings[users]
        pos_item_emb = embeddings[pos_items]
        neg_item_emb = embeddings[neg_items]
        
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        if weights is not None:
            # Temporal Modeling: Apply time-based decay weights
            loss = loss * weights
        return torch.mean(loss)

    def contrastive_loss(self, z1, z2, temp=0.1):
        # Contrastive Learning: InfoNCE loss between two augmented views
        z1, z2 = F.normalize(z1), F.normalize(z2)
        pos_score = torch.exp(torch.sum(z1 * z2, dim=1) / temp)
        all_score = torch.exp(torch.mm(z1, z2.t()) / temp).sum(dim=1)
        return -torch.log(pos_score / all_score).mean()

# --- 2. EVALUATION HELPERS ---
def evaluate_metrics(embeddings, num_users, test_df, top_k=10):
    user_emb = embeddings[:num_users]
    item_emb = embeddings[num_users:]
    
    # Optimization: Map users to their ground truth items
    test_user_items = test_df.groupby('user')['item'].apply(set).to_dict()
    test_users = sorted(test_user_items.keys())
    test_u_tensor = torch.tensor(test_users, dtype=torch.long, device=embeddings.device)
    
    # Vectorized scoring for all test users in one batch
    test_u_emb = user_emb[test_u_tensor]
    all_scores = torch.mm(test_u_emb, item_emb.t())
    all_top_k = torch.topk(all_scores, top_k, dim=1).indices.cpu().numpy()
    
    metrics = {'Hit Rate': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'NDCG': 0.0}
    
    for i, user in enumerate(test_users):
        gt_items = test_user_items[user]
        pred_items = all_top_k[i]
        
        hits = [1 if item in gt_items else 0 for item in pred_items]
        num_hits = sum(hits)
        
        if num_hits > 0:
            metrics['Hit Rate'] += 1
            
        metrics['Precision'] += num_hits / top_k
        metrics['Recall'] += num_hits / len(gt_items)
        
        # Calculate NDCG
        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1 / np.log2(idx + 2) for idx in range(min(top_k, len(gt_items))))
        if idcg > 0:
            metrics['NDCG'] += dcg / idcg

    num_eval_users = len(test_users)
    if num_eval_users > 0:
        for k in metrics:
            metrics[k] /= num_eval_users
            
    return metrics

def evaluate_popularity_metrics(train_df, test_df, top_k=10):
    popular_items = train_df['item'].value_counts().index[:top_k].tolist()
    test_user_items = test_df.groupby('user')['item'].apply(set).to_dict()
    
    metrics = {'Hit Rate': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'NDCG': 0.0}
    test_users = list(test_user_items.keys())
    
    for user in test_users:
        gt_items = test_user_items[user]
        pred_items = popular_items 
        
        hits = [1 if item in gt_items else 0 for item in pred_items]
        num_hits = sum(hits)
        
        if num_hits > 0:
            metrics['Hit Rate'] += 1
            
        metrics['Precision'] += num_hits / top_k
        metrics['Recall'] += num_hits / len(gt_items)
        
        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1 / np.log2(idx + 2) for idx in range(min(top_k, len(gt_items))))
        if idcg > 0:
            metrics['NDCG'] += dcg / idcg

    num_eval_users = len(test_users)
    if num_eval_users > 0:
        for k in metrics:
            metrics[k] /= num_eval_users
            
    return metrics

# --- 3. DATA PREPARATION ---
filename = "Online Retail.xlsx"
if not os.path.exists(filename):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    response = requests.get(url)
    with open(filename, 'wb') as f: f.write(response.content)

df = pd.read_excel(filename)
df = df.dropna(subset=['CustomerID']).copy()

# Data Minimization: Filter frequent items (count > 5)
item_counts = df['StockCode'].value_counts()
df = df[df['StockCode'].isin(item_counts[item_counts > 5].index)].copy()

df['user'] = df['CustomerID'].astype('category').cat.codes
df['item'] = df['StockCode'].astype('category').cat.codes

item_map = df[['item', 'Description']].drop_duplicates('item').set_index('item')['Description'].to_dict()

# Temporal Modeling: Calculate decay weights
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
max_date = df['InvoiceDate'].max()
df['temp_weight'] = np.exp(-0.001 * (max_date - df['InvoiceDate']).dt.days)

num_users = df['user'].nunique()
num_items = df['item'].nunique()
num_nodes = num_users + num_items

# --- 4. EXECUTION LOOP ---
results = {}
pop_results = {}
models_data = {}

# Experimental Setup
normal_pool_df = df.sample(frac=0.2, random_state=42)
train_normal_df, test_df = train_test_split(normal_pool_df, test_size=0.2, random_state=42)
train_ultra_df = train_normal_df.sample(frac=0.05, random_state=42) 

# 1. Train Normal Model
print(f"\n--- Training Normal Model ---")
edge_index_normal = torch.tensor(train_normal_df[['user', 'item']].values.T, dtype=torch.long)
edge_index_normal[1] += num_users 
weights_normal = torch.tensor(train_normal_df['temp_weight'].values, dtype=torch.float)

model_normal = PASGRec(num_nodes, num_users)
optimizer_normal = torch.optim.Adam(model_normal.parameters(), lr=0.01)

loss_history_normal = []
for epoch in range(100):
    model_normal.train()
    optimizer_normal.zero_grad()
    
    ei1, _ = dropout_edge(edge_index_normal, p=0.2)
    ei2, _ = dropout_edge(edge_index_normal, p=0.2)
    
    embeddings = model_normal(edge_index_normal, inject_noise=True)
    z1 = model_normal(ei1, inject_noise=False)
    z2 = model_normal(ei2, inject_noise=False)
    
    loss_bpr = model_normal.bpr_loss(embeddings, edge_index_normal, num_users, num_nodes, weights=weights_normal)
    loss_cl = model_normal.contrastive_loss(z1, z2)
    loss = loss_bpr + 0.1 * loss_cl
    
    loss.backward()
    optimizer_normal.step()
    loss_history_normal.append(loss.item())
    if epoch % 20 == 0: print(f"Normal Epoch {epoch}: Loss {loss.item():.4f}")

# Evaluate Normal
model_normal.eval()
with torch.no_grad():
    embeddings_final_normal = model_normal(edge_index_normal, inject_noise=False)

results['Normal'] = evaluate_metrics(embeddings_final_normal, num_users, test_df, top_k=10)
pop_results['Normal'] = evaluate_popularity_metrics(train_normal_df, test_df, top_k=10)
models_data['Normal'] = (embeddings_final_normal, train_normal_df)

print("\n--- Normal Model Evaluation ---")
for metric, val in results['Normal'].items():
    print(f"PASGRec {metric}@10: {val:.4f}")
print(f"Popularity Hit Rate@10: {pop_results['Normal']['Hit Rate']:.4f}")

# 2. Train Ultra Sparse Model
print(f"\n--- Training Ultra Sparse Model (Warm Started) ---")
edge_index_ultra = torch.tensor(train_ultra_df[['user', 'item']].values.T, dtype=torch.long)
edge_index_ultra[1] += num_users 
weights_ultra = torch.tensor(train_ultra_df['temp_weight'].values, dtype=torch.float)

model_ultra = PASGRec(num_nodes, num_users)
model_ultra.load_state_dict(model_normal.state_dict())
optimizer_ultra = torch.optim.Adam(model_ultra.parameters(), lr=0.001)

loss_history_ultra = []
for epoch in range(150): 
    model_ultra.train()
    optimizer_ultra.zero_grad()
    
    ei1, _ = dropout_edge(edge_index_ultra, p=0.2)
    ei2, _ = dropout_edge(edge_index_ultra, p=0.2)
    
    embeddings_ultra = model_ultra(edge_index_ultra, inject_noise=True)
    z1 = model_ultra(ei1, inject_noise=False)
    z2 = model_ultra(ei2, inject_noise=False)
    
    loss_bpr = model_ultra.bpr_loss(embeddings_ultra, edge_index_ultra, num_users, num_nodes, weights=weights_ultra)
    loss_cl = model_ultra.contrastive_loss(z1, z2)
    loss = loss_bpr + 0.01 * loss_cl
    
    loss.backward()
    optimizer_ultra.step()
    loss_history_ultra.append(loss.item())
    if epoch % 20 == 0: print(f"Ultra Sparse Epoch {epoch}: Loss {loss.item():.4f}")

# Evaluate Ultra Sparse
model_ultra.eval()
with torch.no_grad():
    embeddings_final_ultra = model_ultra(edge_index_ultra, inject_noise=False)

results['Ultra Sparse'] = evaluate_metrics(embeddings_final_ultra, num_users, test_df, top_k=10)
pop_results['Ultra Sparse'] = evaluate_popularity_metrics(train_ultra_df, test_df, top_k=10)
models_data['Ultra Sparse'] = (embeddings_final_ultra, train_ultra_df)

print("\n--- Ultra Sparse Model Evaluation ---")
for metric, val in results['Ultra Sparse'].items():
    print(f"PASGRec {metric}@10: {val:.4f}")
print(f"Popularity Hit Rate@10: {pop_results['Ultra Sparse']['Hit Rate']:.4f}")

# --- 5. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss Curves
ax1.plot(loss_history_normal, label='Normal (20% Data)', color='blue', linewidth=2)
ax1.plot(loss_history_ultra, label='Ultra Sparse (1% Data)', color='orange', linestyle='--', linewidth=2)
ax1.set_title('Model Convergence (BPR Loss)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Hit Rate Comparison
labels = list(results.keys()) 
x = np.arange(len(labels))
width = 0.35

lightgcn_hit_rates = [results[label]['Hit Rate'] for label in labels]
pop_hit_rates = [pop_results[label]['Hit Rate'] for label in labels]

ax2.bar(x - width/2, lightgcn_hit_rates, width, label='PASGRec', color='tab:blue')
ax2.bar(x + width/2, pop_hit_rates, width, label='Popularity Baseline', color='tab:orange')

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel('Hit Rate @ 10')
ax2.set_title('Hit Rate Comparison')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(lightgcn_hit_rates):
    ax2.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center')
for i, v in enumerate(pop_hit_rates):
    ax2.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()
print("\n[Chart saved as comparison.png]")

# --- 6. RECOMMENDATION STABILITY COMPARISON ---
print("\n" + "="*60)
print("STABILITY CHECK: GENERATING RECOMMENDATIONS (OPTIMIZED)")
print("="*60)

# sample_user = int(train_ultra_df['user'].iloc[0])
sample_user = int(random.choice(train_ultra_df['user'].unique()))
recs_store = {}

with torch.no_grad():
    for name in ['Normal', 'Ultra Sparse']:
        emb, _ = models_data[name]
        u_emb = emb[:num_users]
        i_emb = emb[num_users:]
        
        scores = u_emb[sample_user] @ i_emb.T
        top_indices = torch.topk(scores, 10).indices.tolist()
        recs_store[name] = top_indices
        
        print(f"\n{name} Model Top 5 for User {sample_user}:")
        for idx in top_indices[:5]:
            print(f" - {item_map.get(idx, 'Unknown Item')} (ID: {idx})")

matches = set(recs_store['Normal']).intersection(set(recs_store['Ultra Sparse']))
print(f"\nMatching Items: {list(matches)}")
print(f"Match Count: {len(matches)} out of 10")
if len(matches) < 2:
    print("Note: Low overlap usually indicates the 1% model needs more epochs to converge on sparse signals.")
print("="*60)