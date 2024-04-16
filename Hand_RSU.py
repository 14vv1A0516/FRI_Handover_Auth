import socket
import random # import randint
import time
from hashlib import sha256 
import pyexcel as pe
import datetime
import threading
import collections 
from hashlib import sha256 
from ecies.utils import generate_eth_key
from ecies import encrypt, decrypt
import datetime
from ecdsa import SigningKey, SECP256k1, VerifyingKey

class Present:
    def __init__(self, key, rounds=32):
        """Create a PRESENT cipher object

        key:    the key as a 128-bit or 80-bit rawstring
        rounds: the number of rounds as an integer, 32 by default
        """
        self.rounds = rounds
        if len(key) * 8 == 80:
            self.roundkeys = generateRoundkeys80(string2number(key), self.rounds)
        elif len(key) * 8 == 128:
            self.roundkeys = generateRoundkeys128(string2number(key), self.rounds)
        else:
            raise ValueError ("Key must be a 128-bit or 80-bit rawstring")

    def present_encrypt(self, block):
        """Encrypt 1 block (8 bytes)

        Input:  plaintext block as raw string
        Output: ciphertext block as raw string
        """
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[i])
            state = sBoxLayer(state)
            state = pLayer(state)
        cipher = addRoundKey(state, self.roundkeys[-1])
        return number2string_N(cipher, 8)

    def present_decrypt(self, block):
        """Decrypt 1 block (8 bytes)

        Input:  ciphertext block as raw string
        Output: plaintext block as raw string
        """
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[-i - 1])
            state = pLayer_dec(state)
            state = sBoxLayer_dec(state)
        decipher = addRoundKey(state, self.roundkeys[0])
        # print ("Plain text  is ", decipher, "and type is ", type(decipher))
        return number2string_N(decipher, 8)

    def get_block_size(self):
        return 8

# 0   1   2   3   4   5   6   7   8   9   a   b   c   d   e   f
Sbox = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
Sbox_inv = [Sbox.index(x) for x in range(16)]
PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
PBox_inv = [PBox.index(x) for x in range(64)]
import codecs

def generateRoundkeys80(key, rounds):
    """Generate the roundkeys for a 80-bit key

    Input:
            key:    the key as a 80-bit integer
            rounds: the number of rounds as an integer
    Output: list of 64-bit roundkeys as integers"""
    roundkeys = []
    for i in range(1, rounds + 1):  # (K1 ... K32)
        # rawkey: used in comments to show what happens at bitlevel
        # rawKey[0:64]
        roundkeys.append(key >> 16)
        # 1. Shift
        # rawKey[19:len(rawKey)]+rawKey[0:19]
        key = ((key & (2 ** 19 - 1)) << 61) + (key >> 19)
        # 2. SBox
        # rawKey[76:80] = S(rawKey[76:80])
        key = (Sbox[key >> 76] << 76) + (key & (2 ** 76 - 1))
        #3. Salt
        #rawKey[15:20] ^ i
        key ^= i << 15
    print ("80 bit round keys are ", roundkeys)
    return roundkeys

def generateRoundkeys128(key, rounds):
    """Generate the roundkeys for a 128-bit key

    Input:
            key:    the key as a 128-bit integer
            rounds: the number of rounds as an integer
    Output: list of 64-bit roundkeys as integers"""
    roundkeys = []
    for i in range(1, rounds + 1):  # (K1 ... K32)
        # rawkey: used in comments to show what happens at bitlevel
        roundkeys.append(key >> 64)
        # 1. Shift
        key = ((key & (2 ** 67 - 1)) << 61) + (key >> 67)
        # 2. SBox
        key = (Sbox[key >> 124] << 124) + (Sbox[(key >> 120) & 0xF] << 120) + (key & (2 ** 120 - 1))
        # 3. Salt
        # rawKey[62:67] ^ i
        key ^= i << 62
    # print ("128 bit round keys are ", roundkeys)
    return roundkeys


def addRoundKey(state, roundkey):
    return state ^ roundkey


def sBoxLayer(state):
    """SBox function for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""

    output = 0
    for i in range(16):
        output += Sbox[( state >> (i * 4)) & 0xF] << (i * 4)
    return output


def sBoxLayer_dec(state):
    """Inverse SBox function for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(16):
        output += Sbox_inv[( state >> (i * 4)) & 0xF] << (i * 4)
    return output


def pLayer(state):
    """Permutation layer for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox[i]
    return output


def pLayer_dec(state):
    """Permutation layer for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox_inv[i]
    return output


def string2number(i):
    """ Convert a string to a number

    Input: string (big-endian)
    Output: long or integer
    """
    return int(i, 16)


def number2string_N(i, N):
    """Convert a number to a string of fixed size

    i: long or integer
    N: length of string
    Output: string (big-endian)
    """
    s = '%0*x' % (N * 2, i)
    # print ("s is ", s)  
    # return s.decode('hex')
    return str(s)


def _test():
    import doctest
    doctest.testmod()

# Modular arithmetic ##########################################################

def inverse_mod(k, p):
    """Returns the inverse of k modulo p.

    This function returns the only integer x such that (x * k) % p == 1.

    k must be non-zero and p must be a prime.
    """
    if k == 0:
        raise ZeroDivisionError('division by zero')

    if k < 0:
        # k ** -1 = p - (-k) ** -1  (mod p)
        return p - inverse_mod(-k, p)

    # Extended Euclidean algorithm.
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = p, k

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    gcd, x, y = old_r, old_s, old_t

    assert gcd == 1
    assert (k * x) % p == 1

    return x % p

# Functions that work on curve points #########################################

def is_on_curve(point):
    """Returns True if the given point lies on the elliptic curve."""
    if point is None:
        # None represents the point at infinity.
        return True

    x, y = point

    return (y * y - x * x * x - curve.a * x - curve.b) % curve.p == 0


def point_neg(point):
    """Returns -point."""
    assert is_on_curve(point)

    if point is None:
        # -0 = 0
        return None

    x, y = point
    result = (x, -y % curve.p)

    assert is_on_curve(result)

    return result


def point_add(point1, point2):
    """Returns the result of point1 + point2 according to the group law."""
    assert is_on_curve(point1)
    assert is_on_curve(point2)

    if point1 is None:
        # 0 + point2 = point2
        return point2
    if point2 is None:
        # point1 + 0 = point1
        return point1

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2 and y1 != y2:
        # point1 + (-point1) = 0
        return None

    if x1 == x2:
        # This is the case point1 == point2.
        m = (3 * x1 * x1 + curve.a) * inverse_mod(2 * y1, curve.p)
    else:
        # This is the case point1 != point2.
        m = (y1 - y2) * inverse_mod(x1 - x2, curve.p)

    x3 = m * m - x1 - x2
    y3 = y1 + m * (x3 - x1)
    result = (x3 % curve.p,
              -y3 % curve.p)

    assert is_on_curve(result)

    return result


def scalar_mult(k, point):
    """Returns k * point computed using the double and point_add algorithm."""
    assert is_on_curve(point)

    if k % curve.n == 0 or point is None:
        return None

    if k < 0:
        # k * point = -k * (-point)
        return scalar_mult(-k, point_neg(point))

    result = None
    addend = point

    while k:
        if k & 1:
            # Add.
            result = point_add(result, addend)

        # Double.
        addend = point_add(addend, addend)

        k >>= 1

    assert is_on_curve(result)

    return result

# Keypair generation and ECDHE ################################################

def make_keypair():
    """Generates a random private-public key pair."""
    private_key = random.randrange(1, curve.n)
    public_key = scalar_mult(private_key, curve.g)

    return private_key, public_key

def convertTuple_str(tup):
        # initialize an empty string
    dh_str = ''
    for item in tup:
        dh_str = dh_str + str(item) + ","
    return dh_str

def encrypt_data (cipher, plain_text) :

    encrypted_1 = []
    plain_text = plain_text.encode().hex()
    if len(plain_text) > 16 :
        splitted_pt = [plain_text[i:i+16] for i  in range(0, len(plain_text), 16)]
        for each_str in splitted_pt:
            #print ("Encrypting ", each_str)
            encrypted_1.append(cipher.present_encrypt(each_str))
    else :
        encrypted_1 = cipher.present_encrypt(plain_text)
    
    return listToString(encrypted_1)

def decrypt_data (cipher, enc_text) :

    decrypted_1 = []
    splitted_pt = [str(i) for i in enc_text.split(',')]
    for each_str in splitted_pt:
        #print ("Decrypting ", each_str)
        decrypted_1.append(bytes.fromhex(cipher.present_decrypt(each_str)).decode())
    decrypt_temp = ""
    decrypted_1 = decrypt_temp.join(decrypted_1)
    decrypted_1 = decrypted_1.rstrip('\x00').replace('\x00', '')
    
    return decrypted_1

def listToString(s):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += str(ele)
        str1 += ","
    str1 = str1[:len(str1)-1]
    # return string
    return str1

def get_timestamp() :
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    return ts

from hashlib import sha256

def Ver_merkle_path(auth_path, root_hash):

    #print ("Root hash is ", root_hash)
    skip_count = 0

    for key, value in auth_path.items():
        #print(f"Key: {key}, Value: {value}")

        if skip_count >= 3:
            #print ("Into If part ****")
            
            if key[1] == "l" :
                prev_hash = sha256(auth_path[key].encode('utf-8') + prev_hash.encode('utf-8')).hexdigest()
                #print ("11 Prev hash is ", prev_hash)

            elif key[1] == "r" :
                prev_hash = sha256(prev_hash.encode('utf-8') + auth_path[key].encode('utf-8')).hexdigest()
                #print ("22 Prev hash is ", prev_hash)

        else:
            #print ("Into else ---")
            if key[1] == "l" :
                value1 = auth_path[key]
            elif key[1] == "r" :
                value2 = auth_path[key]

            skip_count += 1

            if skip_count == 2 :
                prev_hash = sha256(value1.encode('utf-8') + value2.encode('utf-8')).hexdigest()
                #print ("33 Prev hash is ", prev_hash)
                skip_count += 1

    #print ("Final Prev hash is ", prev_hash)

    if prev_hash == root_hash :
        return 1
    else :
        return 0

# Example usage

def handle_client(veh_conn, SDN_socket) :
    # Create a TCP/IP socket with SDN
    print ("CLient address is ", veh_conn, SDN_socket)
    
    auth_req_VKpubkK = veh_conn.recv(1024).decode()  # receive ( Hand_Req, VKpubK, T1 ) from Veh
    rsu_key_start_latency = time.time ()
    start1_comp_time = time.time ()

    values = [str(i) for i in auth_req_VKpubkK.split(',')]
    T1 = float (values[2] )

    if values[0] == 'H2' and get_timestamp() - T1 < 4:

        print ("Received Authentication request ")
        veh_pub_key = values[1]
        T2 = str(get_timestamp ())
        #rsu_rand = random.randint (100, 100000)
        msg1 = TA_signature.hex() + "#"+ SDN_ID + "#" + RSU_ID + "#"+ RSU_pub_key + "#"+ HSDNrsu + "#"+ T2

        msg1 = msg1.encode('utf-8')
        msg1 = encrypt(veh_pub_key, msg1)

        end1_comp_time = time.time ()
        key_comp_time = end1_comp_time - start1_comp_time

        veh_conn.send(msg1) # send E { TA_sign, SDN_ID, RSU_ID, RSUpubK, HSDNrsu, T2 } to Veh

        data = veh_conn.recv(1024) # Recv E { rVR, Vx, T3 } from Veh
        start2_comp_time = time.time ()

        data = decrypt(RSU_pri_key, data)
        data = data.decode()

        sep_vals = [str(i) for i in data.split('#')]
        rVR = sep_vals[0]
        veh_dh_open_key = sep_vals[1] # receive Vx of veh
        T3 = float (sep_vals[2])

        if get_timestamp () - T3 < 4 :
            values = [str(i) for i in veh_dh_open_key.split(',')]
            veh_dh_open_key = (int(values[0]), int(values[1]))

            RSU_dh_pri_key, dh_open_key = make_keypair()
            dh_open_key = convertTuple_str(dh_open_key)
            T4 = str(get_timestamp () )

            msg2 = rVR + "#"+ dh_open_key + "#"+ T4 

            msg2 = encrypt(veh_pub_key, msg2.encode('utf-8'))
            end2_comp_time = time.time ()

            key_comp_time += end2_comp_time - start2_comp_time

            veh_conn.send(msg2) #  Send E { rVR, Vy, T4 } to Veh
            start3_comp_time = time.time ()

            s1 = scalar_mult(RSU_dh_pri_key, veh_dh_open_key)

            final_int = s1[0] ^ s1[1]

            hash_of_key = sha256(str(final_int).encode())
            final_symmetric_key = hash_of_key.hexdigest()

            end3_comp_time = time.time ()

            key_comp_time += end3_comp_time - start3_comp_time

            rsu_key_end_latency = time.time ()
            rsu_key_exchange_latency = rsu_key_end_latency - rsu_key_start_latency

            print ("\n***** RSU key exchange latency is ", rsu_key_exchange_latency )

            #print ("\nFinal shared key is ", hash_of_key.hexdigest(), "and type is ", type(hash_of_key.hexdigest()))

            # -------- Got final ECDH key ------------------------

            key = final_symmetric_key[:16] # abcdef0123456789abcdef"# .decode('hex')
            
            cipher = Present(key)

            print ("PRESENT Cipher key : ", key)

            # --------- Auth and Ryu comm starts here --------------
        
            RSU_hand_sheet1 = pe.get_sheet (file_name= "STK_Hand_RSU_details.xlsx")

            Hand_Req = veh_conn.recv(1024).decode('utf') # Recv E (VPID, NVIDnew, rA, H2, T1) from Veh
            #print ("\n  Hand_Req from Veh is ", Hand_Req)
            
            Hand_Req = decrypt_data(cipher, Hand_Req)

            SDN_socket.send (Hand_Req.encode('utf'))  # Send (VPID, NVIDnew, rA, H2, T1) to SDNC

            i_ti_f_x_root_hash = SDN_socket.recv(1024).decode('utf')  # recv (Root f(x), NVIDnew, ti, i_val, rH, T2, Root f*(w^2i) ) from SDNC
            start1_comp_time = time.time ()

            i_ti_NVIDnew_f_x_root_hash = [i for i in i_ti_f_x_root_hash.split(',')]
            
            f_x_root_hash = i_ti_NVIDnew_f_x_root_hash[0]
            NVIDnew = i_ti_NVIDnew_f_x_root_hash[1]
            ti = i_ti_NVIDnew_f_x_root_hash[2]
            i_val = i_ti_NVIDnew_f_x_root_hash[3]
            rH_recv = i_ti_NVIDnew_f_x_root_hash [4] 
            T2 = i_ti_NVIDnew_f_x_root_hash [5]
            f_star_w_2i_root_hash = i_ti_NVIDnew_f_x_root_hash[6]

            #print ("f_x_root_hash is ", f_x_root_hash)
            #print ("f_star_w_2i_root_hash is ", f_star_w_2i_root_hash)
            
            NVIDnew_ti_i_val_rH_T2 =  NVIDnew +","+ ti +","+ i_val +","+ rH_recv +","+ T2     # Send (NVIDnew, ti, i_val, rH, T2) to Veh
            NVIDnew_ti_i_val_rH_T2 = encrypt_data (cipher, NVIDnew_ti_i_val_rH_T2)
            end1_comp_time = time.time ()

            comp_time = end1_comp_time - start1_comp_time

            veh_conn.send (NVIDnew_ti_i_val_rH_T2.encode('utf')) # sending E (Root f(x), NVIDnew, ti, i_val, rH, T2, Root f*(w^2i)) to Veh
            proof_pi_rH_T3 = veh_conn.recv (1024).decode('utf') # Recv ( ABC_proof + ","+ auth_path_for_ti + ","+ rH + ","+ T3 ) from Veh

            start2_comp_time = time.time ()
            proof_pi_rH_T3 = decrypt_data(cipher, proof_pi_rH_T3)

            ABC_authpath_ti_rH_T3 = [i for i in proof_pi_rH_T3.split('&')]

            ABC = ABC_authpath_ti_rH_T3[0] # Y1, Y2, Y3 values
            Authpath_ti = eval(ABC_authpath_ti_rH_T3[1])

            rH_recv = ABC_authpath_ti_rH_T3[2]
            T3 = float (ABC_authpath_ti_rH_T3[3])

            #print ("\nAuth path for ti = ", ti, " is ", Authpath_ti, " and type : ", type(Authpath_ti))
            if get_timestamp () - T3 < 4  :

                # ABC_proof = [int(i) for i in Auth_path_proof.split(',')]
                if ti == "0" :
                    #print ("\nVer merkle path for f(w^i) ")
                    merkle_ver_status = Ver_merkle_path (Authpath_ti, f_x_root_hash )
                elif ti == "1" :
                    #print ("\nVer merkle path for fstar(w^2i) ")
                    merkle_ver_status = Ver_merkle_path (Authpath_ti, f_star_w_2i_root_hash )

                #print ("\nMerkle Status is ", merkle_ver_status)

                if merkle_ver_status == 1 :
                    print ("--------- Merkle path Verified SUCCESSFUL ---------")
                    
                    T3 = str (get_timestamp ())

                    ABC_rH_T3 = ABC +"&"+ str(rH_recv) +"&"+ T3
                    end2_comp_time = time.time ()
                    comp_time += end2_comp_time - start2_comp_time

                    SDN_socket.send (ABC_rH_T3.encode('utf')) # sending ( ABC, rH_recv, T3) to SDNC
                    VKi_rH_status_T4 = SDN_socket.recv(1024).decode('utf')  # recv ( VKi, rH, Hand_Status, T4) from SDNC

                    VKi_rH_status_T4 = encrypt_data (cipher, VKi_rH_status_T4)
                    veh_conn.send (VKi_rH_status_T4.encode('utf'))

                    VKi_rH_status_T4 = [str(i) for i in VKi_rH_status_T4.split('&')]
                    VKi = VKi_rH_status_T4[0]

                    print ("Handover Comp Time at RSU is ", comp_time)
                    print ("---- Handover Auth SUCCESS at RSU ------- for ", NVIDnew, "\n ==================== ")

                    RSU_hand_sheet1.row += [VKi, key_comp_time, rsu_key_exchange_latency, comp_time]
                    RSU_hand_sheet1.save_as ("STK_Hand_RSU_details.xlsx")

                else :
                    print ("Auth Path FAILED -------")
            else :
                print ("T3 Timestamp Failed .....")
        else :
            print ("Key T3 Timestamp Failed .....")
    else :
        print ("H2 Handover Request Failed .....")


EllipticCurve = collections.namedtuple('EllipticCurve', 'name p a b g n h')

curve = EllipticCurve(
    'secp256k1',
    # Field characteristic.
    p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f,
    # Curve coefficients.
    a=0,
    b=7,
    # Base point.
    g=(0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
       0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8),
    # Subgroup order.
    n=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141,
    # Subgroup cofactor.
    h=1,
)

SDN_ID = "SDNC1"
RSU_ID ="RSU11"  

print ("********** TA *******\n")
TA_key_obj = generate_eth_key()

#TA_pri_key = 0x545b3aee8f1f1447a50b25390d765a6b83470673363fd8ed62db17d848d8fa03 # TA_key_obj.to_hex ()
#TA_pub_key = 0x5ad96a3a37b57c1e8a082692cda46f230ef45d99e780a6d017a4bf50ab48b191df7314de991a2b44a4888cdcba9f39622f12752cd2445d50c7d704b04d777741 #TA_key_obj.public_key.to_hex ()

#print ("TA pri key is ", TA_pri_key)
#print ("TA pub key is ", TA_pub_key)

RSU_reg_rand = 95793 # random.randint(100, 100000)
#print ("RSU rand no is ", RSU_reg_rand)

RSU_reg_msg = 'SDNC1RSU1195793' # SDN_ID + RSU_ID + str(RSU_reg_rand)
#print ("RSU reg msg is ", RSU_reg_msg)

TA_sign_key = 'e58782f261677cf23b8f2dd374ce93b840737501fb0e38ff6b38cd307e03e871' # SigningKey.generate(curve=SECP256k1)
TA_ver_key = 'd92d4d388c6963b9c15a9820f3f7e3289a6d084d2fdcc3c27bd34ab2df5f4f3de29b0281273473bdba8a2159a097101b304a26cd69d623c35d2a0d5ba649bcb5' #TA_sign_key.verifying_key

print ("TA signing key is ", type(TA_sign_key))
print ("TA verification key is ", type(TA_ver_key))

TA_ver_key_bytes = bytes.fromhex(TA_ver_key)
#print ("**** TA ver key in bytes is ", TA_ver_key_bytes)
TA_ver_key = VerifyingKey.from_string(TA_ver_key_bytes, curve=SECP256k1)
HSDNrsu = sha256(SDN_ID.encode('utf-8') + RSU_ID.encode('utf-8') + str(RSU_reg_rand).encode('utf-8')).hexdigest()

# print ("Pre signature is ", HSDNrsu)
TA_sign_key_bytes = bytes.fromhex (TA_sign_key)
TA_sign_key = SigningKey.from_string(TA_sign_key_bytes, curve=SECP256k1)

TA_signature = TA_sign_key.sign(HSDNrsu.encode()) 
print ("\nTA signature is ", TA_signature)

RSU_key_obj = generate_eth_key()
RSU_pri_key = RSU_key_obj.to_hex() # secret
RSU_pub_key = RSU_key_obj.public_key.to_hex() # format(True)
'''
RSU_pri_key = '0x422973407f0a622178b9eadc3876db9585b781fe8f2642a44860793e5815c648'
RSU_pub_key = '0xf944042434cb4cc5ada05a1b97c572be870f81bbacaaec3a8f2a4964074b14afcd0129cab0f595e441e8f45226a49dd902925730f964e83fdaa667d539dc1229'
'''
print ("\nRSU priv key is ", RSU_pri_key) 
print ("RSU pub key is ", RSU_pub_key)


host = "11.0.0.100" # socket.gethostname()
print("RSU IP is ", host)
print ("-------------------")
port = 6011  # initiate port no above 1024
server_socket = socket.socket()  # get instance
server_socket.bind((host, port))  # bind host address and port together
server_socket.listen(40) 

ryu_host = "10.13.4.78" # ryu ctrler gethostname()
# ryu_port = 8883  # socket server port number
ryu_port = [8881, 8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8880, 8891, 8892, 8893, 8894, 8895, 8896, 8897, 8898, 8899, 8890, 8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808, 8809, 8800]   # random.randint(2000, 10000) # n
ryu_socket = [None] * 35

print ("Got Ryu host ......")
ryu_socket[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[2] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[3] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[4] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[5] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[6] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[7] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[8] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[9] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate

ryu_socket[10] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[11] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[12] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[13] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[14] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[15] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[16] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[17] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[18] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[19] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate

ryu_socket[20] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[21] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[22] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[23] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[24] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[25] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[26] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[27] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[28] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[29] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate
ryu_socket[30] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # instantiate

# ryu_socket.connect((ryu_host, ryu_port))  # connect to the server
i = 0

print ("ENtering WHile loop")
while True :
    veh_conn, client_address = server_socket.accept()
   
    #print ("Ryu -RSU conn port ", ryu_port, " for veh addr ", client_address)
    
    print ("Conn to port ", str(ryu_port[i]))
    ryu_socket[i].connect((ryu_host, ryu_port[i]))  # connect to the serve
    
    #print ("\nRecvd conn from veh ", veh_conn, client_address)
    #print_lock.acquire() # n
    #start_new_thread(handle_client, (veh_conn,)) # n

    client_thread = threading.Thread (target=handle_client, args= (veh_conn, ryu_socket[i]))
    # (client_socket, client_address, abi, acct_address, sc_address, private_key, mydb, mycursor))
    client_thread.start()
    i = i + 1



