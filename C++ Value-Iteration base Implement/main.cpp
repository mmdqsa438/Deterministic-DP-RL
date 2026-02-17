#include <iostream>
#include "DeterministicGridWorld.hpp"


int main( int argc, char* argv[] ){

    // Logging
    std::printf("\n> Test Deterministic Grid world Implementation in C++\n");
    std::printf("> Going to initialize Standard world in cost [0.0]\n");

    // Code
    GridWorld gw1 = standard_GW();

    // Logging
    std::printf("> Standard Grid world -rows: %d -cols: %d\n",gw1.get_rowsize(),gw1.get_colsize());
    std::printf("> Going to initialize Standard world in cost [-0.1]\n");

    // Code 
    GridWorld gw2 = standard_GW(-0.1);
    std::vector<std::vector<std::string>> action_space = {
        {"D","R"},          // [0,0]
        {"R","L"},          // [0,1]
        {"R","L","D"},      // [0,2]
        {},                 // [0,3]
        {"U","D"},          // [1,0]
        {},                 // [1,1]
        {"R","U","D"},      // [1,2]
        {},                 // [1,3]
        {"U","R"},          // [2,0]
        {"R","L"},          // [2,1]
        {"U","R","L"},      // [2,2]
        {"L","U"}           // [2,3]
    };
    std::vector<float> reward_space = { -0.1,-0.1,-0.1,+1.0,-0.1,0.0,-0.1,-1.0,-0.1,-0.1,-0.1,-0.1 };
    gw2.set_config(action_space,reward_space);

    std::printf("> setting Action space & rewards\n");

    int itr = 0;
    int rows = gw2.get_rowsize();
    int cols = gw2.get_colsize();
    std::vector<std::vector<std::string>> all_actions = gw2.get_AllActions();

    // Logging 
    std::printf("\n-------[ Action Space ]-------\n");
    for( auto act_spc : all_actions ){
        std::printf("> State<%d,%d> Action-space: ", itr/cols, itr%cols );
        itr++ ;
        for( std::string action : act_spc ){
            std::cout << action << " ";
        }
        std::printf("\n");
    }
    std::printf("\n> Finish Successfully!\n");
    return 0;
}
