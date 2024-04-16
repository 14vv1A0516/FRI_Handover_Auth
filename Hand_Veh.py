import socket
import sys 
import random # import randint
from hashlib import sha256
from typing import List, Dict
import hashlib
import pyexcel as pe 
from math import floor
import datetime
import sys 
from ecies.utils import generate_eth_key
from ecies import encrypt, decrypt
import time
from ecdsa import SECP256k1, VerifyingKey
import collections

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
        #print ("cipher text  is ", cipher, "and type is ", type(cipher))
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

def generateRoundkeys80(key, rounds): # PRESENT cipher funct
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

def generateRoundkeys128(key, rounds):  # PRESENT cipher funct
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


def addRoundKey(state, roundkey): # PRESENT cipher funct
    return state ^ roundkey


def sBoxLayer(state): # PRESENT cipher funct
    """SBox function for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""

    output = 0
    for i in range(16):
        output += Sbox[( state >> (i * 4)) & 0xF] << (i * 4)
    return output


def sBoxLayer_dec(state): # PRESENT cipher funct
    """Inverse SBox function for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(16):
        output += Sbox_inv[( state >> (i * 4)) & 0xF] << (i * 4)
    return output


def pLayer(state):  # PRESENT cipher funct
    """Permutation layer for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox[i]
    return output


def pLayer_dec(state):  # PRESENT cipher funct
    """Permutation layer for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox_inv[i]
    return output


def string2number(i): # PRESENT cipher funct
    """ Convert a string to a number

    Input: string (big-endian)
    Output: long or integer
    """
    return int(i, 16)


def number2string_N(i, N): # PRESENT cipher funct
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

# =======================PRESENT cipher Over ===============
def convertTuple_str(tup):  #  General funct
    # initialize an empty string
    dh_str = ''
    for item in tup:
        dh_str = dh_str + str(item) + ","
    return dh_str


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

def inverse_mod(k, p): # ECC funct
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

def is_on_curve(point):  # ECC funct
    """Returns True if the given point lies on the elliptic curve."""
    if point is None:
        # None represents the point at infinity.
        return True

    x, y = point

    return (y * y - x * x * x - curve.a * x - curve.b) % curve.p == 0


def point_neg(point):  # ECC funct
    """Returns -point."""
    assert is_on_curve(point)

    if point is None:
        # -0 = 0
        return None

    x, y = point
    result = (x, -y % curve.p)

    assert is_on_curve(result)

    return result

def point_add(point1, point2):  # ECC funct
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

def scalar_mult(k, point):   # ECC funct
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

def make_keypair(): # ECC funct
    """Generates a random private-public key pair."""
    private_key = random.randrange(1, curve.n) 
    public_key = scalar_mult(private_key, curve.g)

    return private_key, public_key

def encrypt_data (cipher, plain_text) :
    encrypted_1 = []
    plain_text = plain_text.encode().hex()
    if len(plain_text) > 16 :
        splitted_pt = [plain_text[i:i+16] for i  in range(0, len(plain_text), 16)]
        for each_str in splitted_pt:
            encrypted_1.append(cipher.present_encrypt(each_str))
    else :
        encrypted_1 = cipher.present_encrypt(plain_text)
    
    return listToString(encrypted_1)

def decrypt_data (cipher, enc_text) :

    decrypted_1 = []
    splitted_pt = [str(i) for i in enc_text.split(',')]
    for each_str in splitted_pt:
        decrypted_1.append(bytes.fromhex(cipher.present_decrypt(each_str) ).decode())
    decrypt_temp = ""
    decrypted_1 = decrypt_temp.join(decrypted_1)
    decrypted_1 = decrypted_1.rstrip('\x00').replace('\x00', '')
    
    return decrypted_1

def get_timestamp() :
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    return ts

def xor_sha_strings(s,t): 
    s = bytes.fromhex(s)
    t = bytes.fromhex(t) 

    res_bytes = bytes(a^b for a,b in zip(s,t))
    return res_bytes.hex()

def listToString(s): 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += str(ele)
        str1 += ","
    str1 = str1[:len(str1)-1]
    return str1

def sha256_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

class Node:
    def __init__(self, left, right, value: str, content, is_copied=False) -> None:
        self.left: Node = left
        self.right: Node = right
        self.value = value
        self.content = content
        self.is_copied = is_copied
         
    @staticmethod
    def hash(val: str) -> str:
        return hashlib.sha256(val.encode('utf-8')).hexdigest()
 
    def __str__(self):
        return (str(self.value))
 
    def copy(self):
        """
        class copy function
        """
        return Node(self.left, self.right, self.value, self.content, True)
       
class MerkleTree:
    def __init__(self, values: List[str]) -> None:
        self.__buildTree(values)
 
    def __buildTree(self, values: List[str]) -> None:
 
        leaves: List[Node] = [Node(None, None, Node.hash(str(e)), str(e)) for e in values]
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1].copy())  # duplicate last elem if odd number of elements
        self.root: Node = self.__buildTreeRec(leaves)
 
    def __buildTreeRec(self, nodes: List[Node]) -> Node:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1].copy())  # duplicate last elem if odd number of elements
        half: int = len(nodes) // 2
 
        if len(nodes) == 2:
            return Node(nodes[0], nodes[1], Node.hash(nodes[0].value + nodes[1].value), nodes[0].content+"+"+nodes[1].content)
 
        left: Node = self.__buildTreeRec(nodes[:half])
        right: Node = self.__buildTreeRec(nodes[half:])
        value: str = Node.hash(left.value + right.value)
        content: str = f'{left.content}+{right.content}'
        return Node(left, right, value, content)
 
    def printTree(self) -> None:
        self.__printTreeRec(self.root)
         
    def __printTreeRec(self, node: Node) -> None:
        if node != None:
            if node.left != None:
                print("Left: "+str(node.left))
                print("Right: "+str(node.right))
            else:
                print("Input")
                 
            if node.is_copied:
                print('(Padding)')
            print("Value: "+str(node.value))
            print("Content: "+str(node.content))
            print("")
            self.__printTreeRec(node.left)
            self.__printTreeRec(node.right)
 
    def getRootHash(self) -> str: 
        return self.root.value

    def inorderTraversal(self, node: Node) -> None:
        if node:
            self.inorderTraversal(node.left)
            print(node.value)
            self.inorderTraversal(node.right)
    
    def getAuthenticationPath(self, value: str, i_val) -> Dict[int, str]:
        path = {}
    
        def findNode(node: Node, value: str, depth: int) -> bool:
            if node is None:
                return False
            elif node.value == value:
                if i_val % 2 == 0:
                    #print ("1depth : ", depth, "Node : ", node.value)
                    path[str(depth)+"l"] = node.value
                else :
                    #print ("1depth : ", depth, "Node : ", node.value)
                    path[str(depth)+"r"] = node.value
                return True
            else:
                if node.left and findNode(node.left, value, depth + 1):

                    #print ("2depth : ", depth, "Node : ", node.right.value)
                    path[str(depth)+"r"] = node.right.value
                    return True
                elif node.right and findNode(node.right, value, depth + 1):

                    path[str(depth)+"l"] = node.left.value
                    #print ("3depth : ", depth, "Node : ", node.left.value)
                    return True
                return False 

        findNode(self.root, value, 0)
        #print ("\n4 Path is  ", path)
        path[str(len(path))+ "z"] = self.root.value  # Add the root hash at the end of the path with the maximum depth
          
        #print ("\n5 depth : ", len(path), "Node : ", self.root.value)

        return path
    
def mixmerkletree(f_w_i) -> None:
    #print("Inputs: ")
    #print(*f_w_i, sep=" | ")
    #print("")
    mtree = MerkleTree(f_w_i)
    #print("Root Hash: "+ mtree.getRootHash()+"\n")
    #mtree.printTree()
    #print("\nInorder Traversal:\n")
    #mtree.inorderTraversal(mtree.root)
    return mtree.getRootHash(), mtree

prime_field = 17
N = 16

TA_ver_key = 'd92d4d388c6963b9c15a9820f3f7e3289a6d084d2fdcc3c27bd34ab2df5f4f3de29b0281273473bdba8a2159a097101b304a26cd69d623c35d2a0d5ba649bcb5'
TA_ver_key_bytes = bytes.fromhex (TA_ver_key)
TA_ver_key = VerifyingKey.from_string(TA_ver_key_bytes, curve=SECP256k1)

veh_key_obj = generate_eth_key()
veh_pri_key = veh_key_obj.to_hex() # secret
veh_pub_key = veh_key_obj.public_key.to_hex() # format(True)

print ("Veh priv key is ", veh_pri_key)
print ("Veh pub key is ", veh_pub_key, "\n")

# ------------ Polynomial generation and send t, degree ------
auth_sheet1 = pe.get_sheet (file_name= "STK_Auth_Veh_details.xlsx")
reg_sheet1 = pe.get_sheet (file_name= "STK_Reg_Veh_details.xlsx")

T1 = str(get_timestamp ())
auth_req_VKpubkK = 'H2,' + veh_pub_key +","+ T1 

# Create a TCP/IP socket
# Create a TCP/IP socket
host = "11.0.0.100" # socket.gethostname()
print("Host IP is ", host)
port = 6011  # initiate port no above 1024

veh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # instantiate

flag = 0
while flag == 0 :
    try :
        veh_socket.connect((host, port))  # connect to the RSU
        flag = 1
    except :
        print ("Failed to connect ...")

print ("------Socket Connected-------------------------\n")

veh_socket.send(auth_req_VKpubkK.encode('utf')) # Send ( Auth_Req, VKpubK, T1 ) to RSU
get_sign = veh_socket.recv(1024)# .decode()  # Recv E { TA_sign, SDN_ID, RSU_ID, RSUpubK, HSDNrsu, T2 } from RSU

key_start1_comp_time = time.time ()

get_sign = decrypt(veh_pri_key, get_sign) # .encode('utf-8'))
#print ("\nDecrypted get sign is ", get_sign)
# --- verify signature
print ("------Socket 000 ....")

get_sign = get_sign.decode()
print ("------Socket 111 ....")
values = [str(i) for i in get_sign.split('#')] # TA_signature, SDN_ID, RSU_ID, RSU_pub_key, HSDNrsu, T2
#print ("\nRcvd signature as ", values)

TA_signature = bytes.fromhex(values[0]) # hex to bytes
SDN_ID = values[1]
RSU_ID = values[2]
RSU_pub_key = values[3]
HSDNrsu_recv = values[4]
T2 = float (values[5])
print ("------Socket 222 ....")

#print ("TA ver key is ", TA_ver_key)
#print ("\nTA signture is ", TA_signature)

# RSU_pub_key = RSU_pub_key.encode().hex()

is_valid = TA_ver_key.verify(TA_signature, HSDNrsu_recv.encode())

print ("\n Is valid ? ", is_valid)
# verify_signature(TA_pub_key, bytes(RSU_reg_msg, 'utf-8'), TA_signature)
if is_valid == True and get_timestamp() - T2 < 4 :

    veh_dh_pri_key, dh_open_key = make_keypair()

    dh_open_key = convertTuple_str(dh_open_key) # Vx

    rVR = random.randint(100, 100000)
    T3 = str(get_timestamp () )
    msg2 = str(rVR) + "#"+ dh_open_key + "#"+ T3
    
    msg2 = encrypt(RSU_pub_key, msg2.encode('utf-8'))

    key_end1_comp_time = time.time ()
    key_comp_time = key_end1_comp_time - key_start1_comp_time

    veh_socket.send(msg2)  # send ( rVR, Vx, T3) to RSU
    data = veh_socket.recv(1024) # Recv  E { rVR, Vy, T4 } from RSU

    key_start2_comp_time = time.time ()

    data = decrypt (veh_pri_key, data)
    data = data.decode()

    values = [str(i) for i in data.split('#')]
    T4 = float (values[2])

    if values[0] == str(rVR) and get_timestamp () - T4 < 4 :
        print ("rVR matched, Diffie helman starts ...")
        values = [str(i) for i in values[1].split(',')]

        RSU_dh_open_key = (int(values[0]), int(values[1]))

        s1 = scalar_mult(veh_dh_pri_key, RSU_dh_open_key)
        #print ("----\n", convertTuple_str(s1))

        final_int = s1[0] ^ s1[1]

        hash_of_key = hashlib.sha256(str(final_int).encode())
        #print (type(s1[0]))
        final_symmetric_key = hash_of_key.hexdigest()
        #print ("Final shared key is ", final_symmetric_key, "and type is ", type(final_symmetric_key))

        key_end2_comp_time = time.time ()

        key_comp_time +=  key_end2_comp_time - key_start2_comp_time
        
        # -------- Got final ECDH key ------------------------
                                                                                                                                                                                                                                     
           
        key = final_symmetric_key[:16] #"0123456789123456" # .decode('hex')

        cipher = Present(key)

        print ("PRESENT Cipher key : ", key)
        print ("*******************************************\n")

        print ("Handover Authentication starts now ---------------\n")
        # ------ Actual Authentication starts now ---------------

        auth_sheet1 = pe.get_sheet (file_name= "STK_Auth_Veh_details.xlsx")
        hand_sheet1 = pe.get_sheet (file_name= "STK_Hand_Veh_details.xlsx")
        reg_sheet1 = pe.get_sheet (file_name= "STK_Reg_Veh_details.xlsx" )

        VID = sys.argv[1]
        PID = sys.argv[2]
        
        VPID = sha256(VID.encode('utf-8') + PID.encode('utf-8')).hexdigest()
        hand_auth_flag = 0

        for row in reg_sheet1 : # Get ( VID, PID, VPID, NVID, f(w^i), fe(w^2i), fo(w^2i), Veh_Reg_comp_time )
            if row[2] == VPID :
                f_w_i = [int(i) for i in row[4].split(',')]
                hand_auth_flag = 1 
                print ("  VPID : ", VPID, "match found ...")
                break

        if hand_auth_flag == 1 :
            print ("f(w^i) : ", f_w_i )
            f_x_w_i_root_hash, f_x_w_i_mtree_obj = mixmerkletree (f_w_i)

            hand_auth_flag = 0

            for row in auth_sheet1 : # Get ( VID, PID, f_star_w_2i, Auth_TxID, rA, NVIDnew, Veh_Init_Auth_comp_time )
                if row[0] == VID :
                    NVIDnew = row[5]
                    print ("f_star_w_2i : ", row[2], " and type is ", type (row[2]))

                    f_star_w_2i = [int(i) for i in row[2].split(',')] 
                    rA = row[4]
                    hand_auth_flag = 1 
                    print ("  VID : ", VID, "match found from Veh Auth sheet ...\n")
                    break

            print ("NVIDnew is ", NVIDnew)

            start1_comp_time = time.time ()

            if hand_auth_flag == 1 :
                Hand_Req = "H2"
                f_star_w_2i_root_hash, f_star_w_2i_mtree_obj = mixmerkletree (f_star_w_2i)
                T1 = str(get_timestamp ())
                Hand_Req = VPID +","+ NVIDnew +","+ rA +","+ Hand_Req +","+ T1 # Send (VPID, NVIDnew, rA, H2, T1) to RSU

                Hand_Req = encrypt_data (cipher, Hand_Req)
                end1_comp_time = time.time ()
                comp_time = end1_comp_time - start1_comp_time

                veh_socket.send (Hand_Req.encode('utf'))
                NVIDnew_ti_i_val_rH_T2 = veh_socket.recv(1024).decode ('utf') # Recv (NVIDnew, ti, i_val, rH, T2) from SDNC

                start2_comp_time = time.time ()

                NVIDnew_ti_i_val_rH_T2 = decrypt_data (cipher, NVIDnew_ti_i_val_rH_T2) 

                values = [str(i) for i in NVIDnew_ti_i_val_rH_T2.split(',')]

                #print ("Values are ", values ) 

                NVIDnew_recv = values [0]
                ti_recv = values [1] 
                i_val = int(values [2])
                rH_recv = values [3]
                T2 = float (values [4])

                if get_timestamp () - T2 < 4 and NVIDnew == NVIDnew_recv :
                    if ti_recv == "0" :
                            auth_path_for_ti = f_x_w_i_mtree_obj.getAuthenticationPath(Node.hash(str(f_w_i[i_val])), i_val)

                    elif ti_recv == "1" :
                            auth_path_for_ti = f_star_w_2i_mtree_obj.getAuthenticationPath(Node.hash(str(f_star_w_2i[i_val])), i_val)

                    #print ("Sending AUth path for tree ti : ", ti_recv, "\n")

                    #print ("\nAuth path for i_val : ", i_val, " is ", auth_path_for_ti)

                    #print ("i_val recvd from RSU : ", i_val, " and type is ", type(i_val))
                    #print ("f_star_w_2i is ", f_star_w_2i)

                    get_f_w_i_val = f_w_i[i_val]
                    get_f_w_N2_i = f_w_i[int(floor(N / 2)) + i_val]
                    get_f_star_w_2i = f_star_w_2i[i_val]
                    
                    ABC_proof = [get_f_w_i_val, get_f_w_N2_i, get_f_star_w_2i]
                    print("\nSending ABC proof is ", ABC_proof)

                    ABC_proof = listToString(ABC_proof)

                    auth_path_for_ti = str(auth_path_for_ti)
                    T3 = str (get_timestamp ())
                    proof_pi_rH = ABC_proof + "&"+ auth_path_for_ti +"&"+ rH_recv + "&"+ T3

                    proof_pi_rH = encrypt_data (cipher, proof_pi_rH)
                    end2_comp_time = time.time ()
                    comp_time += end2_comp_time - start2_comp_time

                    veh_socket.send(proof_pi_rH.encode('utf')) # Send ( ABC_proof + ","+ auth_path_for_ti + ","+ rH + ","+ T3 ) to RSU
                    VKi_rH_status_T4 = veh_socket.recv(1024).decode('utf')  # recv ( VKi, rH, Hand_Status, T4) from SDNC

                    start3_comp_time = time.time ()
                    VKi_rH_status_T4 = decrypt_data (cipher, VKi_rH_status_T4)

                    VKi_rH_status_T4 = [str(i) for i in VKi_rH_status_T4.split('&')]

                    #VKi = VKi_rH_status_T4[0]
                    #rH = VKi_rH_status_T4[1]
                    Hand_status = VKi_rH_status_T4[2] 
                    T4 = float (VKi_rH_status_T4[3] )

                    if Hand_status == "S" and get_timestamp () - T4 < 4 :
                        end3_comp_time = time.time ()
                        comp_time += end3_comp_time - start3_comp_time
                        print ("Veh Comp Time : ", comp_time)
                        print ("--------- Handover Auth SUCCESS --------------")
                        
                        hand_sheet1.row += [VID, key_comp_time, comp_time]
                        hand_sheet1.save_as ("STK_Hand_Veh_details.xlsx")
                    else :
                        print ("Handover Auth Failed -----")
                else :
                    print ("NVIDnew or T2 Mismatch")
            else :
                print ("VID Not matched from Auth excel sheet")
        else :
            print ("VPID Not matched from excel sheet ")
    else :
        print (" Diffie Helman failed ")
else :
    print ("TA Sign Verification Failed  ")       








            