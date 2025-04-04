#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main() {
    std::vector<std::string> rooms{
        "room1",
        "room2",
        "room3",
    };

    json j;

    // key `rooms` and create the json array from the vector:
    j["rooms"] = rooms;
    j["temp"] = 1; // you can put any datatype as value, Json will automatically handle it as far as I know

    std::cout << j << '\n';
}