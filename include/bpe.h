#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class BPEDecoder{
    public:
        std::vector<std::string> vocab;

        bool load(const std::string &vocab_path);
        std::string decode(const std::vector<int>& tokens);
};

class BPEEncoder{
    private:
        std::vector<std::string> vocab;
        std::unordered_map<std::string, int> toi; // token to id
        size_t longest_vocab = 0;

    public:
        bool load(const std::vector<std::string>& vocab);
        const char* encode(const char* text, int* tokens, int max_tokens, int* num_tokens);

};