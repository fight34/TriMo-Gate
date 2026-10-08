import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer
from data_config import load_inputs


df, smiles_list, _target, _xyz_paths = load_inputs()
smiles_list = [i.replace('\xa0', '').replace(' ', '') for i in smiles_list]


all_atom_tokens = []          # List of tokens for each SMILES
parsed_smiles = []            # Flattened list of all tokens

for smi in smiles_list:
    tokens = atomwise_tokenizer(smi)
    all_atom_tokens.append(tokens)
    parsed_smiles.extend(tokens)

# Build token_to_idx
token_to_idx = {'PAD': 0}
unique_tokens = sorted(set(parsed_smiles))
token_to_idx.update({token: idx + 1 for idx, token in enumerate(unique_tokens)})

vocb_size = len(token_to_idx)
max_length = max(len(tokens) for tokens in all_atom_tokens)


# 4. Convert SMILES to index sequences (padding to max_length)
def smiles_to_index(token_list, vocab, length):
    smiles_idx = np.zeros(length, dtype=np.int32)
    # tokens is a list of tokens for a SMILES
    for i, token in enumerate(token_list):
        if i >= length:
            break
        smiles_idx[i] = vocab[token]
    return smiles_idx


indices = np.array([smiles_to_index(tokens, token_to_idx, max_length) for tokens in all_atom_tokens])

print(f'vocabulary_size: {vocb_size}')
print(f'Length of longest SMILES (tokens): {max_length}')
print(f'token_to_index: {token_to_idx}')
print(unique_tokens)
