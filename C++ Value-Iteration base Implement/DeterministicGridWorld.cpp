#include "DeterministicGridWorld.hpp"
#include "algorithm"


// Constractor overloading
//----------------------------
GridWorld::GridWorld( int row, int col, std::pair<int,int> start_state ){

    this->rows = row;
    this->cols = col;
    this->current_state = start_state;

    // initializing States >>>
    for( int r=0; r<row ; r++ ){
        for( int c=0; c<col ; c++ ){
            this->states.push_back( {r,c} );
            this->actions.push_back({});
            this->rewards.push_back(0.0);
        }
    }
}

GridWorld::GridWorld( int row, int col, float cost, std::pair<int,int> start_state ){

    this->rows = row;
    this->cols = col;
    this->current_state = start_state;

    // initializing States >>>
    for( int r=0; r<row ; r++ ){
        for( int c=0; c<col ; c++ ){
            this->states.push_back( {r,c} );
            this->actions.push_back({});
            this->rewards.push_back(cost);
        }
    }
}


// Regular Getters
//---------------------
int GridWorld::get_colsize(void){ return this->cols;}
int GridWorld::get_rowsize(void){ return this->rows;}
std::pair<int,int> GridWorld::get_currentState(void){ return this->current_state;}

std::vector< float > GridWorld::get_AllRewards(void){ return this->rewards;}
std::vector< std::pair<int,int> > GridWorld::get_AllStates(void){ return this->states;}
std::vector< std::vector<std::string> > GridWorld::get_AllActions(void){ return this->actions;}


// Current State Actions getter.
//--------------------------------
std::vector<std::string> GridWorld::get_StateActions(void){
    
    int state_index = this->current_state.first * this->cols + this->current_state.second ;
    return this->actions[state_index];
}

// Environment Transition function: T(s,a)-> r,s'
// In model Base RL learning, transition func will not be used.
//------------------------------------------------
float GridWorld::transition( std::string action ){

    int i = this->current_state.first;
    int j = this->current_state.second;
    int index = i * this->cols + j ;

    std::vector<std::string> avl_actions = this->actions[index];

    if( std::find( avl_actions.begin(), avl_actions.end(), action ) == avl_actions.end() ){
        throw GridWorld::InvalidActionException();
    }

    if ( action == "U" )
        i-- ;
    else if ( action == "D" )
        i++ ;
    else if ( action == "R" )
        j++ ;
    else if ( action == "L" )
        j-- ;
    
    this->current_state = {i,j};
    return this->rewards[ i*this->cols+j ];
}

// Environment Transition Probability function: p(s',r|s,a)-> 0/1
//-----------------------------------------------------------------
float GridWorld::probability( std::pair<int,int> to_state, std::pair<int,int> from_state, std::string action ){

    int i = 0;
    int j = 0;

    if      ( action == "U" ) i-- ;
    else if ( action == "D" ) i++ ;
    else if ( action == "R" ) j++ ;
    else if ( action == "L" ) j-- ;
    else 
        throw GridWorld::InvalidActionException();
    
    int index = from_state.first * this->cols + from_state.second ;
    std::vector< std::string > avl_actions = this->actions[index];
    if( std::find( avl_actions.begin(), avl_actions.end(), action ) == avl_actions.end() ){
        throw GridWorld::InvalidActionException();
    }

    if( (from_state.first + i == to_state.first) && (from_state.second + j == to_state.second) ){
        return 1.0;
    }
    return 0.0;
}


// Some God Mode methods prototype
//----------------------------------
bool GridWorld::is_terminal( std::pair<int,int> state){
    std::vector<std::string> avl_actions = this->actions[ state.first * this->cols + state.second ];
    return avl_actions.empty();
}


// Game-over check-method.
//--------------------------
bool GridWorld::game_over(void){
    std::vector<std::string> avl_actions = this->actions[ this->current_state.first * this->cols + this->current_state.second ];
    return avl_actions.empty();
}


// Updating grid-world's actions & rewards config.
//---------------------------------------------------
void GridWorld::set_config( std::vector< std::vector<std::string> > actions, std::vector< float > rewards){
    this->actions = actions;
    this->rewards = rewards;
}



//  Standard GW creation.
// -------------------------------
GridWorld standard_GW(void){
    int row = 3;
    int col = 4;

    std::pair<int,int> start_state = {2,0};
    GridWorld gw = GridWorld(row, col, start_state);

    return gw;
}

GridWorld standard_GW(float cost){
    int row = 3;
    int col = 4;

    std::pair<int,int> start_state = {2,0};
    GridWorld gw = GridWorld(row, col, cost, start_state);

    return gw;
}

