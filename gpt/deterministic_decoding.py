import torch
from torch.nn import functional as F
import tiktoken
from model import Transformer
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_return_sequences = 5
max_length = 30
seed = 42
set_seed(seed)

model = Transformer.from_pretrained("gpt2")
model.eval()
model.to('cuda')

# prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')
# List to store log probabilities for each sequence
all_log_probs = [[] for _ in range(num_return_sequences)]

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)["logits"] # (B, T, vocab_size)
        logits = logits[:, -1, :]
        # Use a deterministic function instead of softmax
        log_probs = F.log_softmax(logits, dim=-1)
        # Use topk with a stable sorting algorithm
        topk_log_probs, topk_indices = torch.topk(log_probs, 50, dim=-1, sorted=True)
        # Store the top-k log probabilities for each sequence
        for i in range(num_return_sequences):
            all_log_probs[i].append(topk_log_probs[i].cpu().numpy())  
        # Convert log probabilities back to probabilities for sampling
        topk_probs = topk_log_probs.exp() 
        # Use a deterministic sampler
        ix = torch.multinomial(topk_probs, num_samples=1, replacement=True)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# Print the generated text and log probabilities
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"\nSequence {i + 1}:")
    print(">", decoded)
    print("Log probabilities for each step:")
    for step, log_probs in enumerate(all_log_probs[i]):
        print(f"Step {step + 1}: Top 5 log probs = {log_probs[:5]}")