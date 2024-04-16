//SPDX-License-Identifier: MIT

pragma solidity >=0.8.1 <0.9.0;

contract Reg_SC1 
{
    struct reg_struct
    {
       
       string VPID ;
       string root_f_wi ;
       int Cv ;
       int revoc_status ;
    }

    uint256 number ;
    // string HPW ;

    mapping (string => reg_struct) public store_veh_reg ;

    function store_reg_details(string memory NVID, string memory VPID, string memory root_f_wi, int Cv, int  revoc_status) public 
    {
        store_veh_reg[NVID] = reg_struct(VPID, root_f_wi, Cv, revoc_status) ;
    }

    function retrieve_reg_details(string memory NVID) public view returns (reg_struct memory)
    {
        return store_veh_reg[NVID];
    }
}



