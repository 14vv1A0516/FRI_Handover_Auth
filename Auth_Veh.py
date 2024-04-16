import socket
import sys 
import random # import randint
import time
from hashlib import sha256 
from typing import List, Dict
import hashlib
from math import floor 
import datetime
import collections
from ecies.utils import generate_eth_key
from ecies import encrypt, decrypt
from ecdsa import SECP256k1, VerifyingKey
import pyexcel as pe

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
    #print ("80 bit round keys are ", roundkeys)
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

def listToString(s):  # General dunct
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += str(ele)
        str1 += ","
    str1 = str1[:len(str1)-1]
    #print ("str returning is ", str1)
    # return string
    return str1

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

def encrypt_data (cipher, plain_text) : # PRESENT ENCRYPT
    encrypted_1 = []
    plain_text = plain_text.encode().hex()
    if len(plain_text) > 16 :
        splitted_pt = [plain_text[i:i+16] for i  in range(0, len(plain_text), 16)]
        for each_str in splitted_pt:
            encrypted_1.append(cipher.present_encrypt(each_str))
    else :
        encrypted_1 = cipher.present_encrypt(plain_text)
    
    return listToString(encrypted_1)

def decrypt_data (cipher, enc_text) : # PRESENT DECRYPT

    decrypted_1 = []
    splitted_pt = [str(i) for i in enc_text.split(',')]
    for each_str in splitted_pt:
        decrypted_1.append(bytes.fromhex(cipher.present_decrypt(each_str) ).decode())
    decrypt_temp = ""
    decrypted_1 = decrypt_temp.join(decrypted_1)
    decrypted_1 = decrypted_1.rstrip('\x00').replace('\x00', '')
    
    return decrypted_1

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
        
        def findNode(node: Node, depth: int, leaf_index: int) -> bool:
            nonlocal i_val
            if node is None:
                return False
            
            if node.left is None and node.right is None:
                # Check if this is the leaf node and matches the value we're looking for
                if leaf_index == i_val:
                    #print("\nFound leaf node with value:", value)
                    if i_val % 2 == 0:
                        #print("\n1 l depth : ", depth, "& Node : ", node.value)
                        path[str(depth)+"l"] = node.value
                    else:
                        #print("\n1 r depth : ", depth, "& Node : ", node.value)
                        path[str(depth)+"r"] = node.value
                    return True
                else:
                    return False
                
            else:
                if node.left and findNode(node.left, depth + 1, leaf_index * 2):
                    #print("2 r depth : ", depth, "& Node : ", node.right.value)
                    path[str(depth)+"r"] = node.right.value
                    return True
                elif node.right and findNode(node.right, depth + 1, leaf_index * 2 + 1):
                    path[str(depth)+"l"] = node.left.value
                    #print("2 l depth : ", depth, "& Node : ", node.left.value)
                    return True
                return False 

        findNode(self.root, 0, 0)
        path[str(len(path))+ "z"] = self.root.value  # Add the root hash at the end of the path with the maximum depth
        #print("\n3 depth : ", len(path), "& Node : ", self.root.value,"\n")

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

# A utility function to print a polynomial

def printPoly(poly, n):
    # Initialize an empty string to store the polynomial
    polynomial_str = ""

    for i in range(n):
        # Append the coefficient and x term to the string
        if poly[i] != 0:
            polynomial_str += str(poly[i]) + "x^" + str(i) + " + "

    # Remove the trailing " + " from the end of the string
    polynomial_str = polynomial_str[:-3]

    # Print the entire polynomial in a single line
    print(polynomial_str)

def evaluate_polynomial(coefficients, x):
    """
    Evaluate a polynomial given its coefficients at a specific value x.
    
    Parameters:
    - coefficients (list): List of coefficients in descending order (e.g., [a_n, a_{n-1}, ..., a_1, a_0]).
    - x (float or int): Value at which the polynomial should be evaluated.
    
    Returns:
    - result (float or int): Result of evaluating the polynomial at x.
    """
    result = 0
    for i, coef in enumerate(coefficients):
        result += coef * (x ** (len(coefficients) - 1 - i))
    return result

def sha256_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

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
auth_req_VKpubkK = 'A1,' + veh_pub_key +","+ T1 

# Create a TCP/IP socket
host = "10.0.0.100" # socket.gethostname()
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
# Connect the socket to the server's address and port

veh_socket.send(auth_req_VKpubkK.encode('utf')) # Send ( Auth_Req, VKpubK, T1 ) to RSU
get_sign = veh_socket.recv(1024)# .decode()  # Recv E { TA_sign, SDN_ID, RSU_ID, RSUpubK, HSDNrsu, T2 } from RSU

key_start1_comp_time = time.time ()

get_sign = decrypt(veh_pri_key, get_sign) # .encode('utf-8'))
#print ("\nDecrypted get sign is ", get_sign)
# --- verify signature
get_sign = get_sign.decode()

values = [str(i) for i in get_sign.split('#')] # TA_signature, SDN_ID, RSU_ID, RSU_pub_key, HSDNrsu, T2
#print ("\nRcvd signature as ", values)
TA_signature = bytes.fromhex(values[0]) # hex to bytes
SDN_ID = values[1]
RSU_ID = values[2]
RSU_pub_key = values[3]
HSDNrsu_recv = values[4]
T2 = float (values[5])

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
        print (type(s1[0]))
        final_symmetric_key = hash_of_key.hexdigest()
        #print ("Final shared key is ", final_symmetric_key, "and type is ", type(final_symmetric_key))

        key_end2_comp_time = time.time ()

        key_comp_time +=  key_end2_comp_time - key_start2_comp_time
        print ("Key exchange Comp Time : ", key_comp_time)
        # -------- Got final ECDH key ------------------------
                                                                                                                                                                                                                                     
        start_time = time.time()    
        key = final_symmetric_key[:16] #"0123456789123456" # .decode('hex')

        cipher = Present(key)

        print ("PRESENT Cipher key : ", key)
        print ("*******************************************\n")

        print ("Initial Authentication starts now ---------------\n")

        auth_start_latency = time.time ()
        
        VID = sys.argv[1] 
        PID = sys.argv[2]

        VPID = sha256(VID.encode('utf-8') + PID.encode('utf-8')).hexdigest()

        T1 = str(get_timestamp ())

        reg_flag = 0

        for row in reg_sheet1 : # Get ( VID, PID, VPID, NVID, f(w^i), fe(w^2i), fo(w^2i), Veh_Reg_comp_time )
            if row[2] == VPID :
                NVID = row[3]
                f_w_i = row[4]
                fe_w_2i = row[5]
                fo_w_2i = row[6]
                reg_flag = 1 
                print ("  VPID : ", VPID, "match found ...")
                break
            
        start1_comp_time = time.time ()
        if reg_flag == 1 :

            rA = str( random.randint (100, 100000) )
            msg1 = VPID +","+ NVID +","+ str(rA) +","+ "A1," + T1 

            enc_msg1 = encrypt_data (cipher, msg1)

            #print ("ENnc msg1 is ", enc_msg1, " and type is ", type(enc_msg1))
            end1_comp_time = time.time ()
            comp_time = end1_comp_time - start1_comp_time
            
            veh_socket.send (enc_msg1.encode('utf')) # Send E (VPID, NVID, rA, A1, T1) to RSU
            alpha_rA = veh_socket.recv(1024).decode ('utf') # Recv E (alpha, rA, T2) from RSU

            start2_comp_time = time.time ()
            alpha_rA = decrypt_data (cipher, alpha_rA)

            alpha_rA = [str(i) for i in alpha_rA.split(',')]

            #print ("\n dec_alpha_rA : ", alpha_rA[2])
            #print ("\n dec_alpha_rA : ", type(alpha_rA))
            
            #print ("\n T2 : ", alpha_rA[2])
            alpha = int (alpha_rA[0])
            rA_recv = alpha_rA[1]
            T2 = float(alpha_rA[2])

            if get_timestamp () - T2 < 4 and rA == rA_recv :

                #print ("Alpha recvd : ", alpha)

                f_star_w_2i = []
                f_star_len = floor(N/2) # - 1

                f_w_i = [int(i) for i in f_w_i.split(',')]
                fe_w_2i = [int(i) for i in fe_w_2i.split(',')]
                fo_w_2i = [int(i) for i in fo_w_2i.split(',')]

                for i in range(f_star_len) :
                    f_star_w_2i.append( (fe_w_2i[i] + alpha * fo_w_2i[i]) % prime_field ) 

                #print ("f_star_w_2i list is ", f_star_w_2i)
                f_x_w_i_root_hash, f_x_w_i_mtree_obj = mixmerkletree (f_w_i)
                #f_star_w_2i_str = [str(num) for num in f_star_w_2i]

                f_star_w_2i_root_hash, f_star_w_2i_mtree_obj = mixmerkletree (f_star_w_2i)

                #print ("Root of f_star_w_2i : ", f_star_w_2i_root_hash, "\n\n")
                T3 = str(get_timestamp ())
                f_star_root_rA = f_star_w_2i_root_hash +","+ rA_recv +","+ T3

                f_star_root_rA = encrypt_data (cipher, f_star_root_rA)

                end2_comp_time = time.time ()
                comp_time += end2_comp_time - start2_comp_time

                veh_socket.send (f_star_root_rA.encode('utf')) # Send E { f_star_w_2i_root_hash, rA, T3 } to RSU
                ti_i_val_rA = veh_socket.recv(1024).decode('utf') # recv E { ti, i_val, rA, T4 } from RSU

                start3_comp_time = time.time ()

                ti_i_val_rA = decrypt_data (cipher, ti_i_val_rA)
                
                ti_i_val_rA = [str(i) for i in ti_i_val_rA.split(',')]

                ti = ti_i_val_rA[0]
                i_val =  int (ti_i_val_rA[1]) # 3 #
                rA_recv = ti_i_val_rA[2]
                T4 = float (ti_i_val_rA[3])
                '''
                print ("f(w^i) : ", f_w_i)
                print ("fe(w^2i) : ", fe_w_2i)
                print ("fo(w^2i) : ", fo_w_2i)

                print ("f*(w^2i) is ", f_star_w_2i)
                print ("i_val is ", i_val)
                '''
                if get_timestamp () - T4 < 4 and rA == rA_recv :

                    #print ("\nFinding AUth path for ", f_star_w_2i[i_val], " and type is ", type(f_star_w_2i[i_val]))
                    if ti == "0" :
                        auth_path_for_ti = f_x_w_i_mtree_obj.getAuthenticationPath(Node.hash(str(f_w_i[i_val])), i_val)

                    elif ti == "1" :
                        auth_path_for_ti = f_star_w_2i_mtree_obj.getAuthenticationPath(Node.hash(str(f_star_w_2i[i_val])), i_val)

                    #print ("Auth path is ", auth_path_for_ti)

                    #print ("i_val bfr recvd from RSU : ", i_val, " and type is ", type(i_val))

                    #print ("i_val recvd from RSU : ", i_val, " and type is ", type(i_val))
                    #print ("f_star_w_2i is ", f_star_w_2i)

                    get_f_w_i_val = f_w_i[i_val]
                    get_f_w_N2_i = f_w_i[int(floor(N / 2)) + i_val]
                    get_f_star_w_2i = f_star_w_2i[i_val]

                    #print ("get_f_w_i_val is ", get_f_w_i_val)
                    #print ("get_f_w_N2_i is ", get_f_w_N2_i)
                    #print ("get_f_star_w_2i is ", get_f_star_w_2i)

                    #print ("\nf_star_w_2i is ", f_star_w_2i)

                    ABC_proof = [get_f_w_i_val, get_f_w_N2_i, get_f_star_w_2i]
                    #print("\nSending ABC proof is ", ABC_proof)

                    ABC_proof = listToString(ABC_proof)

                    auth_path_for_ti = str(auth_path_for_ti)
                    #Ancestors_list_for_ti = listToString(Ancestors_list_for_ti)

                    T5 = str (get_timestamp ())
                    proof_pi_rA = ABC_proof + "&"+ auth_path_for_ti +"&"+ rA + "&"+ T5

                    proof_pi_rA = encrypt_data (cipher, proof_pi_rA)
                    end3_comp_time = time.time ()
                    comp_time += end3_comp_time - start3_comp_time

                    veh_socket.send(proof_pi_rA.encode('utf')) # Send ( ABC_proof + ","+ auth_path_for_ti + ","+ rA + ","+ T5 ) to RSU
                    TxID_VKi_NVIDnew_rA_status = veh_socket.recv(1024).decode ('utf') # sending (Tx_ID, VKi, NVIDnew, rA, Status, T7) from SDNC

                    start4_comp_time = time.time ()
                    TxID_VKi_NVIDnew_rA_status = decrypt_data (cipher, TxID_VKi_NVIDnew_rA_status)
                    #print (" TxID_VKi_NVIDnew_rA_status :  ", TxID_VKi_NVIDnew_rA_status)
                    TxID_VKi_NVIDnew_rA_status = [str(i) for i in TxID_VKi_NVIDnew_rA_status.split('&')]

                    Auth_Tx_ID = TxID_VKi_NVIDnew_rA_status[0]
                    VKi = TxID_VKi_NVIDnew_rA_status[1]
                    NVIDnew = TxID_VKi_NVIDnew_rA_status[2]
                    rA_recv = TxID_VKi_NVIDnew_rA_status[3]
                    SDN_Auth_status = TxID_VKi_NVIDnew_rA_status[4]
                    T7 = float (TxID_VKi_NVIDnew_rA_status[5])

                    if get_timestamp () - T7 < 4 and rA == rA_recv :
                        #print ("Auth status from RSU : ", SDN_Auth_status)

                        if SDN_Auth_status == "S" :
                            print ("-------- Auth SUCCESS ----------")
                            end4_comp_time = time.time ()
                            comp_time += end4_comp_time - start4_comp_time

                            print ("Total Auth Comp time at Veh is ", comp_time)
                            auth_sheet1.row += [VID, PID, listToString(f_star_w_2i), Auth_Tx_ID, rA, NVIDnew, key_comp_time, comp_time]
                            auth_sheet1.save_as ("STK_Auth_Veh_details.xlsx")

                        else :
                            print ("Auth Failed")
                    else: 
                        print ("rVR and Time stamp T7 Failed")
                else :
                    print ("rVR and Time stamp T4 Failed ")
            else: 
                print ("rVR and Time stamp T2 Failed ")
        else:
            print ("VPID unmatched in Excel Sheet")
    else :
        print ("rVR and Time stamp T4 Failed ")
else :
    print ("TA Sign Verification Failed ")

                    



