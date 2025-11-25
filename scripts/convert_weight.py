import json
import mmap
import numpy as np

from huggingface_hub import hf_hub_download

class WeightMapper:
    '''
    this class downloads the model from huggingface and convert the weight format from safetensors to bin.
    1. model.bin (fp32/fp16 tensors, 32 bit aligned)
    2. model_index.json (tensor metadata for loading)
    '''
    def __init__(self, model_name: str, output_name: str = "model.bin"):
        self.model_name = model_name
        self.model_path = None
        self.output_name = output_name
        self.bin_file = open(self.output_name, "wb")

        self.m = None
        self.metadata = {}

        self.header_offset = 0
        self.bin_offset = 0
        
        # Track layer information for later loading
        self.layer_index = {}

        self._download_model()

    def _download_model(self):
        self.model_path = hf_hub_download(self.model_name, filename = "model.safetensors") # later check how this maps to multiple safetensors
    
    def _calculate_offsets(self, layer_name):
        offs = self.metadata[layer_name]['data_offsets']
        shape = self.metadata[layer_name]['shape']
        dtype = self.metadata[layer_name]['dtype']

        off = self.header_offset + offs[0] # basically the header + starting position
        size = offs[1] - offs[0] # ending - starting
        buffer = self.m[off:off+size]
        
        padded_size = self._store_weights(layer_name, buffer)
        
        # Store layer metadata
        self.layer_index[layer_name] = {
            'offset': self.bin_offset,
            'size': size,
            'padded_size': padded_size,
            'shape': shape,
            'dtype': dtype,
            'transposed': False
        }
        
        return padded_size

    def _calculate_offsets_transposed(self, layer_name):
        offs = self.metadata[layer_name]['data_offsets']
        shape = self.metadata[layer_name]['shape']
        dtype = self.metadata[layer_name]['dtype']

        off = self.header_offset + offs[0] # basically the header + starting position
        size = offs[1] - offs[0] # ending - starting

        w = np.frombuffer(self.m[off: off + size], dtype=np.float32)
        w = w.reshape(self.metadata[layer_name]["shape"])
        w = w.T.copy()
        
        transposed_buffer = w.tobytes()
        padded_size = self._store_weights(layer_name, transposed_buffer)
        
        # Store layer metadata (with transposed shape)
        self.layer_index[layer_name] = {
            'offset': self.bin_offset,
            'size': len(transposed_buffer),
            'padded_size': padded_size,
            'shape': list(reversed(shape)),  # Transposed shape
            'dtype': dtype,
            'transposed': True
        }
        
        return padded_size
    
    def _store_weights(self, layer_name, buff):
        self.bin_file.write(buff)

        buff_size = len(buff)
        padded_size = (buff_size + 31) & (~31) # pad weights to align with 32 bytes
        if padded_size > buff_size:
            print(f"adding padding to {layer_name}")
            self.bin_file.write(bytearray(padded_size - buff_size)) # write the null bytes as padding at the end
        
        return padded_size

    def do_mmap(self):
        with open(self.model_path, 'r') as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as self.m:
                # check here for format - https://github.com/huggingface/safetensors
                header = self.m.read(8)
                n = int.from_bytes(header, byteorder="little")
                metadata_bytes = self.m.read(n) # advance to metadata
                self.metadata = json.loads(metadata_bytes)
                self.header_offset = n + 8 # distance away from header

                self.bin_offset += self._calculate_offsets("wte.weight")
                self.bin_offset += self._calculate_offsets("wpe.weight")
                self.bin_offset += self._calculate_offsets("ln_f.bias")
                self.bin_offset += self._calculate_offsets("ln_f.weight")
                for i in range(12):
                    self.bin_offset += self._calculate_offsets(f"h.{i}.ln_1.bias")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.ln_1.weight")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.attn.c_attn.bias")
                    self.bin_offset += self._calculate_offsets_transposed(f"h.{i}.attn.c_attn.weight")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.attn.c_proj.bias")
                    self.bin_offset += self._calculate_offsets_transposed(f"h.{i}.attn.c_proj.weight")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.ln_2.bias")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.ln_2.weight")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.mlp.c_fc.bias")
                    self.bin_offset += self._calculate_offsets_transposed(f"h.{i}.mlp.c_fc.weight")
                    self.bin_offset += self._calculate_offsets(f"h.{i}.mlp.c_proj.bias")
                    self.bin_offset += self._calculate_offsets_transposed(f"h.{i}.mlp.c_proj.weight")
        
        self.bin_file.close()
        self._save_index()
        return self.bin_offset
    
    def _save_index(self):
        """Save the layer index metadata to a JSON file"""
        index_filename = self.output_name.replace('.bin', '_index.json')
        with open(index_filename, 'w') as f:
            json.dump(self.layer_index, f, indent=2)
        print(f"Saved layer index to {index_filename}")

if __name__ == '__main__':
    mapper = WeightMapper("gpt2")
    total_size = mapper.do_mmap()
    print(f"Total binary size: {total_size} bytes")
    print(f"Layers tracked: {len(mapper.layer_index)}")