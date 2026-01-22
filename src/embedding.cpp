#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string get_tokens(std::string& str) {
    json play = json::parse(str);
    std::cout << play << " " << play["typeCode"] << "\n";
    return play.dump();
}

std::vector<std::string> get_vocab() {
    std::vector<std::string> vocab(100);

    for (int year = 2013; year < 2025; year++){
        std::ifstream inFile(std::format("{}_{}_pbp.bin", year, year+1), std::ios::binary);
        if (!inFile.is_open()){
            std::cerr << "Error opening file" << "\n";
            return vocab;
        }
        
        std::string line;
        while (std::getline(inFile, line)) {

    

    std::ifstream inFile("../data/2013_2014_pbp.bin", std::ios::binary);
    std::string line;
    std::getline(inFile, line);
    std::getline(inFile, line);
    std::cout << line << "\n";
    get_tokens(line);
    return {};
}


int main(){
   get_vocab(); 
}
