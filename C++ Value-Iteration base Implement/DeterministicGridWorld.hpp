#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <utility>

class GridWorld {

private:
    
    int rows, cols;                                  // Grid World Size
    std::pair<int,int> current_state;                // Agent Current State
    std::vector< std::pair<int,int> > states;        // (i,j), (i',j')
    std::vector< std::vector<std::string> > actions; // Actions Peer to peer with it's State   
    std::vector< float > rewards;                    // rewards Peer to peer with it's State

public:

    // Constractor.
    //---------------
    GridWorld( int row, int col, std::pair<int,int> start_state );
    GridWorld( int row, int col, float cost, std::pair<int,int> start_state );
    ~GridWorld(){}

    // Getter - methods
    //--------------------
    int get_colsize(void);
    int get_rowsize(void);
    std::pair<int,int> get_currentState(void);

    std::vector< std::pair<int,int> > get_AllStates(void);
    std::vector< float > get_AllRewards(void);
    std::vector< std::vector<std::string> > get_AllActions(void);


    std::vector<std::string> get_StateActions(void);                // Current State Actions getter.
    float transition( std::string action);                          // Environment Transition function: T(s,a)-> r,s'

    // Full Model - Base methods
    //-----------------------------
    float probability( std::pair<int,int> to_state, std::pair<int,int> from_statem, std::string action );

    bool is_terminal( std::pair<int,int> state);
    bool game_over(void);

    void set_config( std::vector< std::vector<std::string> > actions, std::vector< float > rewards );


    // Inner Classes.
    //-----------------------------
    class InvalidActionException : public std::exception {};

};

// Helper Functions
//--------------------
GridWorld standard_GW(float cost);
GridWorld standard_GW(void);
