from torch import nn
from torch.nn import functional as F
import torch
class HeadAttention(nn.Module):

    def __init__(self, embedding_dim:int , head_size:int, context_len:int , dropout :float):
        super().__init__()
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(self.embedding_dim, self.head_size, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.head_size, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.head_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropput = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))
        
    
    
    def forward(self, x):
        # x shape: (batch_size, context_len, embedding_dim)
        # Project input x into query, key, and value
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores (use scaled dot-product attention)
        # queries shape: (batch_size, context_len, head_size)
        # keys shape: (batch_size, context_len, head_size)
        # Scale dot-product by sqrt(head_size)
        scores = (queries @ keys.transpose(-2, -1)) / (self.head_size ** 0.5)

        scores = scores.masked_fill(self.tril== 0, float('-inf')) # (B, T, T)
        scores = self.softmax(scores) # (B, T, T)
        scores = self.dropput(scores)
        attention = scores @ values  # Apply softmax to get attention weights

        
        return attention



class FeedForward(nn.Module):
    def __init__(self, embedding_dim:int, dropout:float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = 4*embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.dropout(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim:int, context_len:int,  head_size:int, num_heads:int, dropout :float):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.heads = nn.ModuleList([HeadAttention(embedding_dim, head_size, context_len,  dropout) for _ in range(num_heads)])
        self.out = nn.Linear(self.num_heads * self.head_size, self.embedding_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        heads_ouput = [head(x) for head in self.heads]
        heads_ouput_tensor = torch.stack(heads_ouput, dim=3)
        heads_ouput_tensor = heads_ouput_tensor.view(x.shape[0], x.shape[1], self.num_heads * self.head_size)

        return self.dropout(self.out(heads_ouput_tensor))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, context_len ,  head_size, num_heads, dropout ):
        super().__init__()
        head_size = embedding_dim//num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, context_len=context_len, head_size=head_size, num_heads=head_size, dropout=dropout)
        self.feed_forward = FeedForward(embedding_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        attention_output = self.multi_head_attention(x)
        x = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        return self.layer_norm2(x + feed_forward_output)
    


class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size:int , embedding_dim:int, context_len:int, head_size:int, num_heads:int, num_transformer_blocks:int, dropout:float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_len = context_len
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embedding_dim, context_len , head_size, num_heads, dropout) for _ in range(num_transformer_blocks)])
        self.layernorm_final = nn.LayerNorm(embedding_dim) # final layer norm
        self.out = nn.Linear(embedding_dim, vocab_size)

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(context_len, embedding_dim)

    
    def forward(self, idx):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(self.context_len, device=idx.device))
        x = tok_emb + pos_emb
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.layernorm_final(x) # (B,T,C)
        logits = self.out(x) # (B,T,vocab_size)
        return logits





@torch.no_grad()    
def generate(model, tokenizer, prompt_ids, max_new_tokens, context_len,device='cuda', temperature=1.0):
    """
    prompt_ids: (seq_len, ) - starting token IDs
    max_new_tokens: how many new tokens to generate
    temperature: controls randomness (lower = more deterministic)
    """
    model.eval()
    
    # Get pad token ID
    pad_id = tokenizer.token_to_id("<pad>")
    
    # Process input - pad if needed
    if len(prompt_ids) < context_len:
        padded_prompt = prompt_ids + [pad_id] * (context_len - len(prompt_ids))
    else:
        padded_prompt = prompt_ids[-context_len:]
    
    # Convert to tensor and add batch dimension
    idx = torch.tensor([padded_prompt], dtype=torch.long).to(device)
    
    # Print starting text
    print("\n--- Starting with: ---")
    print(f"{tokenizer.decode(prompt_ids)}")
    print("---------------------------------\n")
    
    
    # Generate tokens sequentially
    for _ in range(max_new_tokens):
        # Crop context if it's getting too long
        idx_cond = idx if idx.size(1) <= context_len else idx[:, -context_len:]
        
        # Forward pass to get logits
        logits = model(idx_cond)
        
        # Focus on the last token's prediction
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # Apply temperature to control randomness
        logits = logits / temperature
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # Append the new token to our sequence
        idx = torch.cat((idx, next_idx), dim=1)
        
        # Print the current state
        next_token = tokenizer.decode([next_idx[0].item()], skip_special_tokens=False)
        generated_so_far = tokenizer.decode(idx[0].tolist(), skip_special_tokens=False)
        
        # Check if we've generated a complete poem stanza
        if ("<linebreak>" in next_token) or ("<eos>" in next_token):
            print("\n--- Complete poem generated ---")
            print(generated_so_far)
            print("---------------------------------\n")
            break
        
            
    return idx

    