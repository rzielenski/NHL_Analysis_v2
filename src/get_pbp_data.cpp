#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <curl/curl.h>
#include <string>
#include <tuple>
#include <format>
#include <vector>
#include <thread>

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output){
    size_t total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}

std::tuple<json, long> get_game_pbp(int year, int game_type, int game_num)
{
    //curl -X GET "https://api-web.nhle.com/v1/gamecenter/2023020204/play-by-play"
    CURL* curl = curl_easy_init();
    std::string buf;
    long http_code = 0;
    
    if (!curl) { return {json{}, 0}; }

    
    const std::string url = std::format("https://api-web.nhle.com/v1/gamecenter/{}{:02d}{:04d}/play-by-play", year, game_type, game_num);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "nhl-transformer/1.0 (libcurl)");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);

    CURLcode res = curl_easy_perform(curl);

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK){
        std::cerr << "curl_easy_perform() failed: " << url << curl_easy_strerror(res) << "\n";
        return {json{}, http_code};
    }
    if (http_code != 200) {
        std::cerr << "HTTP " << http_code << " for " << url << "\n";
        if (!buf.empty()){
            std::cerr << "Body head: " << buf.substr(0, 200) << "\n";
        }
        return {json{}, http_code};
    }
    try {
        return {json::parse(buf), http_code};
    } catch (const json::parse_error& e){
        std::cerr << "JSON parse error for " << url << ": " << e.what() << "\n";
        std::cerr << "Body head: " << buf.substr(0, 200) << "\n";
        return {json{}, http_code};
    }
}

void get_year_pbp(int year, std::vector<std::string>& games, int start, int end) {
    for (int game = start; game < end; game++) {
       auto [game_json, http_code] = get_game_pbp(year, 2, game);

       if (http_code == 404) { break; }
       if (http_code != 200) { continue; }
       std::string home = game_json.value("homeTeam", json{}).value("abbrev", "");
       std::string away = game_json.value("awayTeam", json{}).value("abbrev", "");

       std::string res = "home: " + home + " away: " + away;
       if (game_json.contains("plays") && game_json["plays"].is_array()){
           for (const auto& play : game_json["plays"]){
                res += "\n" + play.dump(); 
           }
       }

       if (game - 1 >= 0 && game - 1 < (int)games.size()){
           games[game - 1] = std::move(res);
       }
    } 
}

void save_to_bin(const std::vector<std::string>& data, const std::string& filename){
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile){
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    
    for (const std::string& s : data){
        if (s.empty()) continue;
        outfile.write(s.c_str(), (std::streamsize)s.size());
        outfile.put('\n');
    }
    outfile.close();
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::vector<std::string> all_games(1400);
    for (int year = 2013; year < 2025; year++){    
        {
            std::vector<std::jthread> threads;
            threads.emplace_back([&] { get_year_pbp(year, all_games, 1, 200); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 201, 400); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 401, 600); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 601, 800); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 801, 1000); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 1001, 1200); });
            threads.emplace_back([&] { get_year_pbp(year, all_games, 1201, 1400); });  
        }
        
        save_to_bin(all_games, std::format("../data/{}_{}_pbp.bin", year, year+1));
        std::cout << "Completed " << year << "\n";
        all_games.clear();
    }
    

    curl_global_cleanup();
    return 0;
}

