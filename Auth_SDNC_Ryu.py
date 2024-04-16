import socket
import string 
import random # import randint
import time
from hashlib import sha256
import pyexcel as pe
from math import floor
import datetime
import threading
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls 
import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
from ryu.ofproto import ofproto_v1_3

def get_timestamp() :
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    return ts

def handle_client(RSU_socket) :
        print ("CLint connnnected -------------")
        auth_sheet = pe.get_sheet(file_name="STK_Auth_SDNC_details.xlsx")
        Auth_Req = RSU_socket.recv(1024).decode('utf')  # Send (VPID, NVID, rA, A1, T1) from Veh

        SDN_auth_start1_latency = time.time ()
        start1_comp_time = time.time ()
        values = [str(i) for i in Auth_Req.split(',')]

        VPID = values[0]
        NVID = values[1]
        rA = values[2]
        auth_req = values[3]
        T1 = float (values[4])

        if auth_req == "A1" and get_timestamp() - T1 < 4 :
            
            # reg_nonce = w3.eth.getTransactionCount(acct_address)
            reg_cont_instance = w3.eth.contract(address = reg_sc_address, abi = reg_abi) # creates an instance of init auth contract
    
            auth_nonce = w3.eth.get_transaction_count(acct_address)
            auth_cont_instance = w3.eth.contract(address = auth_sc_address, abi = auth_abi) # creates an instance of handover contract
            SDN_auth_end1_latency = time.time ()
            SDN_auth_latency = SDN_auth_end1_latency - SDN_auth_start1_latency
            
            get_NVID_data = reg_cont_instance.functions.retrieve_reg_details(NVID).call() # run 
            print ("Reg Details for NVID: ", NVID, "from BC is ", get_NVID_data )
            print ("\n Type of BC data : ", type (get_NVID_data))
            SDN_auth_start2_latency = time.time ()
            
            reg_VPID = get_NVID_data[0] 
            f_x_root_hash = get_NVID_data[1] 
            Cv = get_NVID_data[2]
            Rs = get_NVID_data[3]
            
            #check_TID = w3.eth.get_transaction(NVID)
            # print ("check_TID is ", check_TID)
            #tid_BC_hash = check_TID.__dict__["hash"].hex()
            #print ("Tx ID from BC is ", tid_BC_hash)
            
            #f_x_root_hash = "de16e9f44b7773edad4b7e9ce2bdcffe5686028ee135985bdad9c095fce5fb49"
            
            if VPID == reg_VPID : # "8c83b6fbde25b520e6e80497381448abd1100ae15e9cf013e460ad3bdd953602" :# get_NVID_data[0]
                print ("Transaction happened for Reg")
            
                #print ("Reg Detaiget_NVID_datals of HPW ", values[1], " are ", get_NVID_data, "--------------\n")

                i_val = random.randint(0, floor(N/2)-1) # 3 #
                T2 = str(get_timestamp ())
                i_f_x_root_hash = f_x_root_hash + ","+ str(i_val) + ","+ rA + ","+ T2

                #print ("\nSending i_val : ", i_val)
                #print ("Sending i and f(x) root hash : ", i_f_x_root_hash)

                end1_comp_time = time.time ()
                comp_time = end1_comp_time - start1_comp_time 

                RSU_socket.send (i_f_x_root_hash.encode('utf')) # send (Root f(x), i_val, rA, T2) to RSU
                ABC_proof_alpha_T6 = RSU_socket.recv(1024).decode("utf") # Recv (f*(w^2i)_root_hash, ABC, alpha, rA_recv, T6) from RSU

                #print ("ABC_proof_alpha_T6 : ", ABC_proof_alpha_T6)
                ABC_proof_alpha_T6 = [i for i in ABC_proof_alpha_T6.split('&')]

                #print ("ABC_proof_alpha_T6 list : ", ABC_proof_alpha_T6)

                f_star_w_2i_root_hash = ABC_proof_alpha_T6[0]
                ABC = ABC_proof_alpha_T6[1]
                alpha_recv = int (ABC_proof_alpha_T6 [2])
                rA_recv = ABC_proof_alpha_T6 [3]
                T6 = float (ABC_proof_alpha_T6[4])
                '''
                print ("\nRecvd ABC proof : ", ABC)
                print ("Alpha is ", alpha_recv)
                print ("rA recv is ", rA_recv)
                print ("T6 is ", T6)
                '''
                start2_comp_time = time.time ()

                if get_timestamp () - T6 < 4 and rA == rA_recv :
                    ABC_proof_list = [int(i) for i in ABC.split(',')]

                    #print ("Received ABC Y-coordinates are ", ABC_proof_list)

                    # Example usage:
                    x_values = [ (w**i_val) % prime_field, (w**(floor(N/2)+ i_val)) % prime_field] #, alpha 
                    y_values = [ ABC_proof_list[0] , ABC_proof_list[1] ] # , ABC_proof_list[2]
                    x_to_evaluate = alpha_recv

                    print ("A : (", x_values[0], ",", y_values[0], ")")
                    print ("B : (", x_values[1], ",", y_values[1], ")")

                    w_minus_i_mod_p = pow(w, -i_val, prime_field)
                    inv_2_mod_p = pow(2, -1, prime_field)

                    term1 = 1 + alpha_recv * w_minus_i_mod_p 
                    term2 = 1 - alpha_recv * w_minus_i_mod_p

                    y3_for_alpha = ((term1 *  y_values[0] + term2 * y_values[1] ) * inv_2_mod_p) % prime_field
                    print ("\nComputed C : (", alpha_recv, ",", y3_for_alpha, ")")

                    #print(f"The result of Lagrange interpolation at x = {x_to_evaluate} is: {y3_for_alpha}")

                    if y3_for_alpha == ABC_proof_list[2]:
                        print ("---- Lagrange interpolation Ver SUCCESSFUL-----------")

                        rA_auth = random.randint(100, 100000)
                        T7 = get_timestamp ()

                        NVIDnew =  sha256 ( NVID.encode('utf-8') + str(rA_auth).encode('utf-8') + str(T7).encode('utf-8') ).hexdigest()

                        Sn_ki = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
                        As = str ( random.randint(100, 100000) )
                        VIDnew = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))

                        VKi = Sn_ki +","+ As +","+ VIDnew 
                        end2_comp_time = time.time ()
                        comp_time += end2_comp_time - start2_comp_time
                        '''
                        #rand_final = sha256(str(proof).encode('utf-8') + str(t).encode('utf-8') + str(r).encode('utf-8')).hexdigest()
                        print ("Writing new NVID = ", NVIDnew, "and type is ", type(NVIDnew))  
                        print ("Writing VPID = ", VPID, "and type is ", type(VPID))
                        print ("Writing alpha  is ", alpha_recv, "and type is ", type(alpha_recv))
                        print ("Writing f_x_root_hash = ", f_x_root_hash, "and type is ", type(f_x_root_hash))
                        print ("Writing f_star_w_2i_root_hash = ", f_star_w_2i_root_hash, "and type is ", type(f_star_w_2i_root_hash))
                        print ("Writing Cv = ", Cv, "and type is ", type(Cv))
                        print ("Writing Rs = ", Rs, "and type is ", type(Rs))
                        '''
                        SDN_auth_end2_latency = time.time ()
                        SDN_auth_latency += SDN_auth_end2_latency - SDN_auth_start2_latency

                        # NVIDnew,  VPID, alpha, root_f_wi, root_f_w2i, Cv, revoc_status 
    
                        #   str ,   str,    int    ,  str     , str,    int,     int 
                        call_function = auth_cont_instance.functions.store_auth_details(NVIDnew, VPID, alpha_recv, f_x_root_hash, f_star_w_2i_root_hash, Cv, Rs).build_transaction({"chainId": Chain_id, "from": acct_address, "nonce": auth_nonce, "gasPrice": w3.to_wei(2, 'gwei')})
                        #print ("call_function Tx : ", call_function)
                        signed_tx = w3.eth.account.sign_transaction(call_function, private_key = private_key)
                        #print ("sign Tx : ", signed_tx)
                        send_tx = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                        #print ("Send Tx : ", send_tx )
                        
                        tx_receipt = w3.eth.wait_for_transaction_receipt(send_tx)
            
                        Tx_ID = tx_receipt.__dict__["transactionHash"].hex()
                        print("Tx hash is ", Tx_ID)

                        trans = w3.eth.get_transaction (Tx_ID)
            
                        gas_price = trans.gasPrice
                        #print ("GasUsed is ",tx_receipt.gasUsed)
                        tx_fee_in_gwei = tx_receipt.gasUsed * gas_price/1000000000
                        #print ("Total tx fee is ", tx_fee_in_gwei, " gwei")
            
                        SDN_auth_start3_latency = time.time ()
                        T7 = str (get_timestamp ())

                        VKi_NVIDnew_rA_status = Tx_ID +"&"+ VKi +"&"+ NVIDnew +"&"+ rA +"&"+ "S" +"&"+ T7

                        RSU_socket.send (VKi_NVIDnew_rA_status.encode('utf')) # send (Tx_ID, VKi, NVIDnew, rA, Status, T7) to RSU
                        SDN_auth_end3_latency = time.time ()

                        SDN_auth_latency += SDN_auth_end3_latency - SDN_auth_start3_latency

                        #print ("............ AUthentication successful ...")
                        auth_sheet.row += [tx_fee_in_gwei, SDN_auth_latency, comp_time]
                        auth_sheet.save_as ("STK_Auth_SDNC_details.xlsx")
                        
                        print ("Total Auth Comp time at SDNC is ", comp_time)
                        print ("Total Auth Latency at SDNC is ", SDN_auth_latency, "\n===================================")
                    else :
                        print ("------- Ver Failllllll-------- ")
                        RSU_socket.send ("F".encode('utf'))
                else :
                    print ("rA and Tinmestamp T6 match failed")
            else :
                print ("VPID mismatch from Blockchain ")
        else :
            print ("Invalid Auth Req and T1 check failed ")
        
N = 16
prime_field = 17
w = 7

provider_url = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(provider_url))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

print (w3.is_connected())

reg_abi = '[{ "inputs": [ { "internalType": "string", "name": "NVID", "type": "string" }, { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "name": "store_reg_details", "outputs": [], "stateMutability": "nonpayable", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "NVID", "type": "string" } ], "name": "retrieve_reg_details", "outputs": [ { "components": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "internalType": "struct Reg_SC1.reg_struct", "name": "", "type": "tuple" } ], "stateMutability": "view", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "", "type": "string" } ], "name": "store_veh_reg", "outputs": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "stateMutability": "view", "type": "function" } ]'

acct_address = '0x92f3EEc2D42D4219b78B227AA20C791E749Ce64D'

reg_sc_address = '0xBD470eF0D07B112173D9421b872E1BEe1aDa623a'
private_key = '0x671882cf9099ab762529b48b7f9713b833fb7499c9af2585e835952ee63efc41'

auth_abi = '[ { "inputs": [ { "internalType": "string", "name": "NVIDnew", "type": "string" } ], "name": "retrieve_auth_details", "outputs": [ { "components": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "string", "name": "root_f_w2i", "type": "string" },{ "internalType": "int256", "name": "alpha", "type": "int256" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "internalType": "struct Auth_SC2.auth_struct", "name": "", "type": "tuple" } ], "stateMutability": "view", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "NVIDnew", "type": "string" }, { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "int256", "name": "alpha", "type": "int256" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "string", "name": "root_f_w2i", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "name": "store_auth_details", "outputs": [], "stateMutability": "nonpayable", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "", "type": "string" } ], "name": "store_veh_auth", "outputs": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "string", "name": "root_f_w2i", "type": "string" }, { "internalType": "int256", "name": "alpha", "type": "int256" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "stateMutability": "view", "type": "function" } ]'
auth_sc_address = "0xFC5470C2Ee2B32695Cb2CE1D841D63B6d19997Ff"

Chain_id = w3.eth.chain_id

host = "10.13.4.78" # socket.gethostname()

port1 = 8881  # socket server port number
port2 = 8882
port3 = 8883
port4 = 8884
port5 = 8885
port6 = 8886
port7 = 8887
port8 = 8888
port9 = 8889
port10 = 8880
port11 = 8891
port12 = 8892
port13 = 8893
port14 = 8894
port15 = 8895
port16 = 8896
port17 = 8897
port18 = 8898
port19 = 8899
port20 = 8890
port21 = 8801
port22 = 8802
port23 = 8803
port24 = 8804
port25 = 8805
port26 = 8806
port27 = 8807
port28 = 8808
port29 = 8809
port30 = 8800


ryu_scket = [None] * 40

ryu_scket[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[0].bind((host, port1))  # bind host address and port together
ryu_scket[0].listen(20)

ryu_scket[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[1].bind((host, port2))  # bind host address and port together
ryu_scket[1].listen(20)

ryu_scket[2] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[2].bind((host, port3))  # bind host address and port together
ryu_scket[2].listen(20)

ryu_scket[3] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[3].bind((host, port4))  # bind host address and port together
ryu_scket[3].listen(20)

ryu_scket[4] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[4].bind((host, port5))  # bind host address and port together
ryu_scket[4].listen(20)

ryu_scket[5] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[5].bind((host, port6))  # bind host address and port together
ryu_scket[5].listen(20)

ryu_scket[6] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[6].bind((host, port7))  # bind host address and port together
ryu_scket[6].listen(20)

ryu_scket[7] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[7].bind((host, port8))  # bind host address and port together
ryu_scket[7].listen(20)

ryu_scket[8] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[8].bind((host, port9))  # bind host address and port together
ryu_scket[8].listen(20)

ryu_scket[9] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[9].bind((host, port10))  # bind host address and port together
ryu_scket[9].listen(20)

ryu_scket[10] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[10].bind((host, port11))  # bind host address and port together
ryu_scket[10].listen(20)

ryu_scket[11] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[11].bind((host, port12))  # bind host address and port together
ryu_scket[11].listen(20)

ryu_scket[12] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[12].bind((host, port13))  # bind host address and port together
ryu_scket[12].listen(20)

ryu_scket[13] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[13].bind((host, port14))  # bind host address and port together
ryu_scket[13].listen(20)

ryu_scket[14] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[14].bind((host, port15))  # bind host address and port together
ryu_scket[14].listen(20)

ryu_scket[15] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[15].bind((host, port16))  # bind host address and port together
ryu_scket[15].listen(20)

ryu_scket[16] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[16].bind((host, port17))  # bind host address and port together
ryu_scket[16].listen(20)

ryu_scket[17] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[17].bind((host, port18))  # bind host address and port together
ryu_scket[17].listen(20)

ryu_scket[18] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[18].bind((host, port19))  # bind host address and port together
ryu_scket[18].listen(20)

ryu_scket[19] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[19].bind((host, port20))  # bind host address and port together
ryu_scket[19].listen(20)

ryu_scket[20] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[20].bind((host, port21))  # bind host address and port together
ryu_scket[20].listen(20)

ryu_scket[21] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[21].bind((host, port22))  # bind host address and port together
ryu_scket[21].listen(20)

ryu_scket[22] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[22].bind((host, port23))  # bind host address and port together
ryu_scket[22].listen(20)

ryu_scket[23] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[23].bind((host, port24))  # bind host address and port together
ryu_scket[23].listen(20)

ryu_scket[24] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[24].bind((host, port25))  # bind host address and port together
ryu_scket[24].listen(20)

ryu_scket[25] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[25].bind((host, port26))  # bind host address and port together
ryu_scket[25].listen(20)

ryu_scket[26] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[26].bind((host, port27))  # bind host address and port together
ryu_scket[26].listen(20)

ryu_scket[27] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[27].bind((host, port28))  # bind host address and port together
ryu_scket[27].listen(20)

ryu_scket[28] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[28].bind((host, port29))  # bind host address and port together
ryu_scket[28].listen(20)

ryu_scket[29] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_scket[29].bind((host, port30))  # bind host address and port together
ryu_scket[29].listen(20)



i = 0
while True:
    nonce_lock = threading.Lock ()
    print ("Waiting for Vehicle Conn ...on port i =  ",i, ryu_scket[i] )
    print ("===============================")
    connection, client_address = ryu_scket[i].accept()
    veh_thread = threading.Thread(target=handle_client, args=(connection,))
    i = i + 1
    veh_thread.start()
