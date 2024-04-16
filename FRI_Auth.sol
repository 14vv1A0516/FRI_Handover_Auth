//AUth.sol
//-------------------------

//SPDX-License-Identifier: MIT

pragma solidity >=0.8.1 <0.9.0;

contract Auth_SC2
{
    struct auth_struct
    {
       string VPID ;
       string root_f_wi ;
       string root_f_w2i ;
       int alpha ;
       int Cv ;
       int revoc_status ;
    }

    uint256 number ;
    // string HPW ;

    mapping (string => auth_struct) public store_veh_auth ;

    function store_auth_details (string memory NVIDnew, string memory VPID, int alpha, string memory root_f_wi, string memory root_f_w2i, int Cv, int  revoc_status ) public 
    {
        store_veh_auth[NVIDnew] = auth_struct(VPID, root_f_wi, root_f_w2i, alpha, Cv, revoc_status) ;
    }

    function retrieve_auth_details (string memory NVIDnew) public view returns (auth_struct memory)
    {
        return store_veh_auth[NVIDnew];
    }
}
