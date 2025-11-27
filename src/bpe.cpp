#include <iostream>
#include <string>
#include <fstream>

#include "bpe.h"

bool BPEDecoder::load(const std::string &vocab_path) {
    std::ifstream file(vocab_path, std::ios::binary);
    if (!file) return false;

    vocab.clear();

    while (true) {
        unsigned char len;
        if (!file.read((char*)&len, 1))
            break; 

        std::string token(len, '\0'); // token has '\0' * len times
        if (!file.read(&token[0], len))
            break;

        vocab.push_back(token);
    }

    return true;
}

std::string BPEDecoder::decode(const std::vector<int>& tokens) {
    std::string output;

    for (int tok : tokens) {
        if (tok == -1) {
            output += "(?)"; // possible unknown token
            continue;
        }

        if (tok < 0 || tok >= vocab.size())
            break;

        output += vocab[tok];
    }

    return output;
}

//TO-DO encoder check for merges or trie based

bool BPEEncoder::load(const std::vector<std::string>& vocab){
    for(int i = 0; i < vocab.size(); i ++){
        toi[vocab[i]] = i;

        longest_vocab = std::max(longest_vocab, vocab[i].size());
    }
    return true;
}

const char* BPEEncoder::encode(const char* text, int* tokens, int max_tokens, int* num_tokens) {
    *num_tokens = 0;
    
    while (*text && *num_tokens < max_tokens) {
        size_t try_len = std::min(longest_vocab, strlen(text));
        bool found = false;
        
        // search from longest possible down to 1 character
        for (size_t len = try_len; len > 0; len--) {
            std::string candidate(text, len);
            
            auto it = toi.find(candidate);
            if (it != toi.end()) {
                tokens[(*num_tokens)++] = it->second;
                
                text += len; // advance the pointer by len of the match
                
                found = true;
                break;
            }
        }
        
        if (!found) return text;
    }
    
    return text;
}
