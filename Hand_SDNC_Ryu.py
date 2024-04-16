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
from web3 import Web3
from ryu.ofproto import ofproto_v1_3

def get_timestamp() :
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    return ts

def handle_client(RSU_socket) :
    print ("CLint connnnected -------------")
    auth_sheet = pe.get_sheet(file_name="STK_Hand_SDNC_details.xlsx")
    Hand_Req = RSU_socket.recv(1024).decode('utf')  # Recv E (VPID, NVIDnew, rA, H2, T1) from RSU

    hand_start_latency = time.time ()
    start1_comp_time = time.time ()

    values = [str(i) for i in Hand_Req.split(',')]

    VPID = values[0]
    NVIDnew = values[1]
    rH = values[2]
    hand_req = values[3]
    T1 = float (values[4])

    #print ("NVIDnew is ", NVIDnew, " and type : ", type(NVIDnew) )

    if hand_req == "H2" and get_timestamp() - T1 < 4 :
        
        auth_cont_instance = w3.eth.contract(address = auth_sc_address, abi = auth_abi) # creates an instance of init auth contract
    
        get_NVIDnew_data = auth_cont_instance.functions.retrieve_auth_details(NVIDnew).call() # run 
        # VPID,  alpha,  root_f_wi, root_f_w2i,   Cv,   revoc_status
        # str ,   int ,    str    ,    str    ,  int ,    int 

        print ("NVIDnew Details from BC : ", get_NVIDnew_data)

        VPID_BC = get_NVIDnew_data [0]
        f_x_root_hash = get_NVIDnew_data [1]
        f_star_x_root_hash = get_NVIDnew_data [2]
        alpha_BC = get_NVIDnew_data [3]
        Cv = get_NVIDnew_data [4]
        Rs = get_NVIDnew_data [5]


        #check_TID = w3.eth.get_transaction(NVIDnew)
        # print ("check_TID is ", check_TID)
        #tid_BC_hash = check_TID.__dict__["hash"].hex()
        #print ("Tx ID from BC is ", tid_BC_hash) 

        if VPID == VPID_BC and Rs == 0 :
        
            #f_x_root_hash = "de16e9f44b7773edad4b7e9ce2bdcffe5686028ee135985bdad9c095fce5fb49"
            #f_star_x_root_hash = "69df6e29adbcc5e1262afaf851f08516e868bee43ac687f6f93eee01cb545aca"

            ti = random.randint(0, 1) # which merkle tree to verify [f(x) = 0 or f*(x) = 1]
            rH = str( random.randint (100, 100000) )

            i_val = random.randint(0, floor(N/2)-1) # 3 #

            T2 = str(get_timestamp ())
            #print ("f_x_root_hash : ", f_x_root_hash, " and type : ", type(f_x_root_hash))
            #print ("NVIDnew : ", NVIDnew, " and type : ", type(NVIDnew))

            #print ("ti : ", ti, " and type : ", type(ti))
            #print ("i_val : ", i_val, " and type : ", type(i_val))
            #print ("rH : ", rH, " and type : ", type(rH))
            #print ("T2 : ", T2, " and type : ", type(T2))

            #print ("f_star_x_root_hash : ", f_star_x_root_hash, " and type : ", type(f_star_x_root_hash))
            

            i_ti_f_x_root_hash = f_x_root_hash + ","+ NVIDnew +","+ str(ti) +","+ str(i_val) + ","+ rH + ","+ T2 + ","+ f_star_x_root_hash

            end1_comp_time = time.time ()

            comp_time = end1_comp_time - start1_comp_time

            RSU_socket.send (i_ti_f_x_root_hash.encode('utf')) # send (Root f(x), NVIDnew, ti, i_val, rH, T2, f*(w^2i) root hash) to RSU
            ABC_proof_rH_T3 = RSU_socket.recv(1024).decode("utf") # Recv  ( ABC, rH_recv, T3) from RSU

            start2_comp_time = time.time ()
            ABC_proof_rH_T3 = [i for i in ABC_proof_rH_T3.split('&')]

            #print ("ABC_proof_alpha_T6 list : ", ABC_proof_alpha_T6)

            ABC = ABC_proof_rH_T3[0]
            rH_recv = ABC_proof_rH_T3 [1]
            T3 = float (ABC_proof_rH_T3[2])

            #print ("\nRecvd ABC proof : ", ABC)
            #print ("rH recv is ", rH_recv)
            #print ("T3 is ", T3)

            if get_timestamp () - T3 < 4 and rH == rH_recv :
                ABC_proof_list = [int(i) for i in ABC.split(',')]

                #print ("Received ABC Y-coordinates are ", ABC_proof_list)
                # alpha_BC = 3
                # Example usage:
                x_values = [ (w**i_val) % prime_field, (w**(floor(N/2)+ i_val)) % prime_field] #, alpha 
                y_values = [ ABC_proof_list[0] , ABC_proof_list[1] ] # , ABC_proof_list[2]
                x_to_evaluate = alpha_BC

                print ("A : (", x_values[0], ",", y_values[0], ")")
                print ("B : (", x_values[1], ",", y_values[1], ")")

                w_minus_i_mod_p = pow(w, -i_val, prime_field)
                inv_2_mod_p = pow(2, -1, prime_field)

                term1 = 1 + alpha_BC * w_minus_i_mod_p 
                term2 = 1 - alpha_BC * w_minus_i_mod_p

                y3_for_alpha = ((term1 *  y_values[0] + term2 * y_values[1] ) * inv_2_mod_p) % prime_field
                print ("\nComputed C : (", alpha_BC, ",", y3_for_alpha, ")")

                print(f"The result of Lagrange interpolation at x = {x_to_evaluate} is: {y3_for_alpha}")

                if y3_for_alpha == ABC_proof_list[2]:
                    print ("---- Lagrange interpolation Ver SUCCESSFUL-----------")

                    Sn_ki = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
                    As = str ( random.randint(100, 100000) )
                    VIDnew = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))

                    VKi = Sn_ki +","+ As +","+ VIDnew 
                    
                    '''
                    #rand_final = sha256(str(proof).encode('utf-8') + str(t).encode('utf-8') + str(r).encode('utf-8')).hexdigest()
                    print ("Writing new NVID = ", new_NVID, "and type is ", type(new_NVID))  
                    print ("Writing t(x) = ", get_HPW_data[1], "and type is ", type(get_HPW_data[1]))
                    print ("Writing HPW is ", value1[0], "and type is ", type(value1[0]))
                    print ("Writing rand_final = ", rA_auth, "and type is ", type(rA_auth))
                    print ("Writing proof = ", proof, "and type is ", type(proof))
                    print ("Writing Cv = ", get_HPW_data[3], "and type is ", type(get_HPW_data[3]))
                    '''
                    T4 = str (get_timestamp ())

                    VKi_rH_status_T4 = VKi +"&"+ rH +"&"+ "S" +"&"+ T4
                    end2_comp_time = time.time ()
                    comp_time += end2_comp_time - start2_comp_time

                    RSU_socket.send (VKi_rH_status_T4.encode('utf')) # send ( VKi, rH, Hand_Status, T4) to RSU
                    hand_end_latency = time.time ()

                    SDN_hand_latency = hand_end_latency - hand_start_latency

                    print (".... ........ AUthentication successful ...")

                    auth_sheet.row += [SDN_hand_latency, comp_time]
                    auth_sheet.save_as ("STK_Hand_SDNC_details.xlsx")
                            
                    print ("Total Auth Comp time at SDNC is ", comp_time)
                    print ("Total Auth Latency at SDNC is ", SDN_hand_latency, "\n===================================")
                else :
                    print ("------- Lagrange Ver Failllllll-------- ")
                    RSU_socket.send ("F".encode('utf'))
            else :
                print ("BC data mismatch ....")
        else :
            print ("rH and Tinmestamp T3 match failed")
    else :
        print ("Invalid Auth Req and T1 check failed ")

N = 16
prime_field = 17
w = 7

provider_url = "http://127.0.0.1:8545"

w3 = Web3(Web3.HTTPProvider(provider_url))
print (w3.is_connected())

acct_address = '0x11C2f6C6ec01Dca241821Acc93321725aAe8e325'

private_key = '0xaba7fc58ad52484222bebdff433085c0cb3c6ad9310f141adc80ff37175bd954'
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
