import socket 
import random 
from math import floor 
import pyexcel as pe
import time
from hashlib import sha256  
from web3 import Web3
import datetime
from web3.middleware import construct_sign_and_send_raw_middleware
from web3.middleware import geth_poa_middleware
from eth_account import Account
import threading

def get_timestamp() :
    # ct stores current time
    ct = datetime.datetime.now()
    # ts store timestamp of current time
    ts = ct.timestamp()
    return ts

def xor_sha_strings(s,t):
    s = bytes.fromhex(s)
    t = bytes.fromhex(t)

    res_bytes = bytes(a^b for a,b in zip(s,t))
    return res_bytes.hex()

def handle_client(client_socket):
    TA_reg_sheet1 = pe.get_sheet (file_name= "STK_Reg_TA_details.xlsx")
    

    data = client_socket.recv(1024).decode() # Recv ( VID, VPID, TPW, UID, T1) from Veh
    reg_start1_latency = time.time ()
    #print ("Data recivd is ", data)
    start1_comp_time = time.time ()

    values = [str(i) for i in data.split(',')]
    
    print ("Recvd VID and TPW values ", values[3])
    VID = values[0]
    VPID = values[1]
    TPW = values[2]
    UID = values[3]
    T1 = float(values[4])
    Tc = get_timestamp ()

    if Tc - T1 < 4 :
        x = random.randint(2, 102)
        y = random.randint(2, 102)
        temp1 = int(pow(w, x, prime_field))
        temp2 = int(pow(w_, y, prime_field))

        Cv = (temp1 * temp2) % prime_field
        t = time.localtime()

        NTPW = xor_sha_strings(TPW, sha256(str(TPW).encode('utf-8') + str(x).encode('utf-8') + str(t).encode('utf-8')).hexdigest())

        T1 = str(get_timestamp())
        data = VID+ ","+ NTPW  + ","+ T1
        print (" BC transaction Pending --- \n")

        end1_comp_time = time.time ()
        comp_time = end1_comp_time - start1_comp_time 

        print ("Sending Veh_ID  = ", VID, "\n NTPW = ", NTPW, "to Vehicle")

        client_socket.send(data.encode())  # send (VID, NTPW, T2) to the Veh
        data = client_socket.recv(1024).decode() # Recv (NTPW, f(w^i) merkle root hash, T3) from Veh

        start2_comp_time = time.time ()
        values = [str(i) for i in data.split(',')]
        #print ("f_w_i_merkle_root values are ", values)

        f_w_i_merkle_root = values[1] 
        T3 = float (values[2])

        if NTPW == values[0] and get_timestamp() - T3 < 4:

            NVID = xor_sha_strings(NTPW, sha256(str(random.randint(2, 102)).encode('utf-8') + str(random.randint(2, 102)).encode('utf-8')).hexdigest()) 
            #start_time = time.time()

            T4 = str(get_timestamp())
            msg4 = NVID + "," + NTPW+ ","+ T4
            
            end2_comp_time = time.time ()
            comp_time += end2_comp_time - start2_comp_time 

            client_socket.send(msg4.encode())  # send data to the client

            #print ("new_RPW matched\n writing NVID_new = ",NVID_new,"t_eq, degree, Cv into Blockchain")
            
            nonce = w3.eth.getTransactionCount(acct_address)
             # creates an instance of contract
            cont_instance = w3.eth.contract(address = sc_address, abi = reg_abi)
            reg_end1_latency = time.time ()

            reg_latency = reg_end1_latency - reg_start1_latency

            call_function = cont_instance.functions.store_reg_details(NVID, VPID, f_w_i_merkle_root, Cv, 0 ).buildTransaction({"chainId": Chain_id, "from": acct_address, "nonce": nonce, "gasPrice": w3.toWei(2, 'gwei')}) # acct.address for localhost
            # Store NVID, VPID, Cv, Rs, Root{f(w^i)} in SCreg Blockchain
             
            signed_tx = w3.eth.account.sign_transaction(call_function, private_key = private_key)

            send_tx = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            tx_receipt = w3.eth.wait_for_transaction_receipt(send_tx)

            #print ("tx_receipt is ", tx_receipt)

            Tx_ID = tx_receipt.__dict__["transactionHash"].hex()
            print("Tx hash is ", Tx_ID)

            trans = w3.eth.get_transaction (Tx_ID)
            reg_start2_latency = time.time ()
            
            gas_price = trans.gasPrice
            print ("GasUsed is ",tx_receipt.gasUsed)
            #print ("Gas price in wei is ", gas_price)
            tx_fee_in_gwei = tx_receipt.gasUsed * gas_price/1000000000
            print ("Total tx fee is ", tx_fee_in_gwei, " gwei")
            
            print ("Hurrray! Transaction Done Successfully\n ----------------------------------------")
            
        else:
            print("NVID doesn't match, Registration failed")
            
        reg_end2_latency = time.time ()

        reg_latency += reg_end2_latency - reg_start2_latency
        TA_reg_sheet1.row += [VID, UID, str(x), str(y), VPID, reg_latency, comp_time, tx_fee_in_gwei ]
        TA_reg_sheet1.save_as ("STK_Reg_TA_details.xlsx")

        
        print ("Total TA Comp Time is ", comp_time, " sec \n ======================\n \n")

        print ("Total Registration latency is ", reg_latency , " sec \n ======================\n \n")

        client_socket.close () 
    

#provider_url = "https://sepolia.infura.io/v3/f9e641160e574eba873b5fc1e47a9e69" # http://127.0.0.1:7545"
provider_url = "http://127.0.0.1:8545"

w3 = Web3(Web3.HTTPProvider(provider_url)) 

# For localhost
reg_abi = '[{ "inputs": [ { "internalType": "string", "name": "NVID", "type": "string" }, { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "name": "store_reg_details", "outputs": [], "stateMutability": "nonpayable", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "NVID", "type": "string" } ], "name": "retrieve_reg_details", "outputs": [ { "components": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "internalType": "struct Reg_SC1.reg_struct", "name": "", "type": "tuple" } ], "stateMutability": "view", "type": "function" }, { "inputs": [ { "internalType": "string", "name": "", "type": "string" } ], "name": "store_veh_reg", "outputs": [ { "internalType": "string", "name": "VPID", "type": "string" }, { "internalType": "string", "name": "root_f_wi", "type": "string" }, { "internalType": "int256", "name": "Cv", "type": "int256" }, { "internalType": "int256", "name": "revoc_status", "type": "int256" } ], "stateMutability": "view", "type": "function" } ]'
acct_address = "0x11C2f6C6ec01Dca241821Acc93321725aAe8e325"  
sc_address = "0xBD470eF0D07B112173D9421b872E1BEe1aDa623a"
private_key = "0xaba7fc58ad52484222bebdff433085c0cb3c6ad9310f141adc80ff37175bd954"
acct = w3.eth.account.from_key('0xaba7fc58ad52484222bebdff433085c0cb3c6ad9310f141adc80ff37175bd954')
print ("acct is ", acct)

# w3_middleware = w3.middleware_onion.add(construct_sign_and_send_raw_middleware(acct))
w3_middleware = w3.middleware_onion.inject(geth_poa_middleware, layer=0)
print ("w3_middleware obj is ",w3_middleware)

w3.eth.default_account = acct.address
print("Middle ware client version is ", w3.eth.default_account)

print (w3.isConnected())

Chain_id = w3.eth.chain_id 

prime_field = 17
w = 7
w_ = 3

host = "localhost" # socket.gethostname()
print("Host IP is ", host)
port = 6002  # initiate port no above 1024
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # get instance
server_socket.bind((host, port))  # bind host address and port together

server_socket.listen(40)
i =0 

while True :
    client_socket, client_address = server_socket.accept()
    i = i + 1
    client_thread = threading.Thread (target=handle_client, args= (client_socket,))
    client_thread.start()
