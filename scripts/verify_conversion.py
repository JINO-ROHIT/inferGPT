import json
import numpy as np

with open("model/model_index.json", 'r') as f:
    metadata = json.load(f)

all_weights = {}

with open("model/model.bin", 'rb') as f:
    for layer_name, layer_info in metadata.items():
        f.seek(layer_info['offset'])
        data = f.read(layer_info['size'])
        weights = np.frombuffer(data, dtype=np.float32).reshape(layer_info['shape'])
        all_weights[layer_name] = weights
        print(f"Loaded {layer_name}: {weights.shape}")