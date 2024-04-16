import socket 
import string 
import random # import randint
import time
from hashlib import sha256
from typing import List
import hashlib
import datetime
import pyexcel as pe

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
    '''
    def inorderTraversal(self, node: Node) -> None:
        if node:
            self.inorderTraversal(node.left)
            print(node.value)
            self.inorderTraversal(node.right)
    '''
    def getAuthenticationPath(self, value: str) -> List[str]:
        path = []
        def findNode(node: Node, value: str) -> bool:
            if node is None:
                return False
            elif node.value == value:
                path.append(node.value)
                return True
            else:
                if node.left and findNode(node.left, value):
                    path.append(node.right.value)
                    return True
                elif node.right and findNode(node.right, value):
                    path.append(node.left.value)
                    return True
                return False
        findNode(self.root, value)
        path.append(self.root.value)
        return path
    
    def getAncestorslist(self, value: str) -> List[str]:
        path = []
        def findNode(node: Node, value: str) -> bool:
            if node is None:
                return False
            elif node.value == value:
                # path.append(node.value)
                return True
            else:
                if node.left and findNode(node.left, value):
                    path.append(node.left.value)
                    return True
                elif node.right and findNode(node.right, value):
                    path.append(node.right.value)
                    return True
                return False
        findNode(self.root, value)
        path.append(self.root.value)
        return path
    
def mixmerkletree(f_w_i) -> None:
    #print("Inputs: ")
    #print(*f_w_i, sep=" | ")
    #print("")
    mtree = MerkleTree(f_w_i)
    print("Root Hash: "+ mtree.getRootHash()+"\n")
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

# Example usage:
#poly = [2, 0, 5, 0, 1]  # Example polynomial coefficients i.e, const 1st =2 
#degree = len(poly)  # Degree of the polynomial
#printPoly(poly, degree)
'''
def printPoly(poly, n):
    for i in range(n):
        print(poly[i], " ")
        if (i != 0):
            print("x^", i, " ")
        if (i != n - 1):
            print(" + ", " ")
'''
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

prime_field = 17 # w= 7 (generator), F_p field
w = 7
N = 16
ID_size = 7
w_ = 3
reg_sheet1 = pe.get_sheet (file_name= "STK_Reg_Veh_details.xlsx")
# Compute the subgroup
D_subgroup = [1, 7, 15, 3, 4, 11, 9, 12, 16, 10, 2, 14, 13, 6, 8, 5]  # w^i till N = 16, D domain
w_2i = [1, 15, 4, 9, 16, 2, 13, 8] # w^2i till N = 8, D* domain

VID = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(ID_size))
UID = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(ID_size))
print ("The Vehicle ID is: " + VID)

PID = input("Enter password for veh_ID '{0}': ".format(VID))

start1_comp_time = time.time ()

VPID = sha256(VID.encode('utf-8') + PID.encode('utf-8')).hexdigest()

r = random.randint(100, 100000)
t = time.localtime()
T1 = get_timestamp ()

temp_rT1 = sha256(str(r).encode('utf-8') + str(t).encode('utf-8')).hexdigest()

TPW = xor_sha_strings(VPID, temp_rT1)
msg1 = VID+ ','+ VPID +','+ TPW+ ','+ UID+ ','+ str(T1)  # send (VID, VPID, TPW, UID, T1) to TA

end1_comp_time = time.time ()

comp_time = end1_comp_time - start1_comp_time

# Create a TCP/IP socket
TA_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('localhost', 6002)
TA_socket.connect(server_address)
print(f"Connected to {server_address}") 

TA_socket.send (msg1.encode('utf')) # send (VID, TPW, UID, T1) to TA

msg2 = TA_socket.recv(1024).decode()  # receive (VID, NTPW, T2) from Veh

start2_comp_time = time.time ()

values = [str(i) for i in msg2.split(',')]
VID_recv = values[0]
NTPW = values[1]
T2 = float(values[2])
Tc = get_timestamp ()

if Tc - T2 < 4 and VID == VID_recv :
    print ("Received Veh_ID = ", VID, "\nNTPW = ", NTPW, " from TA")

    # ------------ Polynomial generation and send f, degree ------

    f_deg = random.randint(12, 16)
    print("Degree is ", f_deg)

    #print("Polynomial f(x)= ")
    fx_list = []

    for i in range(0, f_deg + 1):
        fx_list.append(random.randint(-100, 100))

    #fx_list = [2, 5, 10, 6, 7, 9, 4, 14]

    f_len = len(fx_list)
    #print ("f list from lowest degree term  is ", f_list)  # [a_n, a_{n-1}, ..., a_1, a_0]
    printPoly(fx_list, f_len)
    fx_list.reverse ()

    print ("Poly f list highest degree term is ", fx_list) # [a_0, a_1, ..., a_{n-1}, a_n]

    f_w_i = []

    for each in D_subgroup :
        result = evaluate_polynomial(fx_list, each) # f_list : [a_n, a_{n-1}, ..., a_1, a_0]
        f_w_i.append(result % prime_field)  

    #print(f"The result of evaluating the polynomial at x = {each} is: {result}")
    print("Subgroup D(w^i): ", D_subgroup)

    f_w_i_str = [str(num) for num in f_w_i]
    f_w_i_root_hash, f_w_i_mtree = mixmerkletree (f_w_i_str)

    # print ("f_w_i_root_hash is ", f_w_i_root_hash)

    T3 = str(get_timestamp())

    end2_comp_time = time.time ()

    comp_time += end2_comp_time - start2_comp_time 

    NTPW_mrkle_root = NTPW + ","+  f_w_i_root_hash +","+ T3
    TA_socket.send (NTPW_mrkle_root.encode('utf')) # send (NTPW, f_w_i_root_hash, T3) to TA Veh

    #print ("Sending NTPW_mrkle_root : ", NTPW_mrkle_root)
    msg2 = TA_socket.recv(1024).decode()  # receive (NVID, NTPW, T4) from Veh

    start3_comp_time = time.time ()
    values = [str(i) for i in msg2.split(',')] 

    NVID = values[0]
    NTPW_recv = values[1]
    T4 = float(values[2])
    if get_timestamp() - T4 < 4 and NTPW == NTPW_recv :

        start2_comp_time = time.time()

        fo_list = []
        fe_list = []

        #print ("f_w_i is ", f_w_i , "\n")

        deg = len(fx_list)
        i = 0

        if deg % 2 == 0: # fx is odd degree
            while i < len(fx_list)-2:
                fo_list.append(fx_list[i])
                i = i + 1
                fe_list.append(fx_list[i])
                i = i + 1
            if i == len(fx_list)-2 :
                fo_list.append(fx_list[i])
                i = i + 1
                fe_list.append(fx_list[i])
        
        elif deg % 2 != 0: # fx is even degree
            while i < len(fx_list)-2:
                fe_list.append(fx_list[i])
                i = i + 1
                if i == len(fx_list)-2 :
                    fo_list.append(fx_list[i])
                    i = i + 1
                    fe_list.append(fx_list[i])
                else:
                    fo_list.append(fx_list[i])
                    i = i + 1

        print ("fe(x) : ", fe_list)
        print ("fo(x) : ", fo_list)

        #if get_timestamp() - float(values[2]) < 4:

        fe_w_2i = []
        fo_w_2i = []

        for each in w_2i :
            result = evaluate_polynomial(fe_list, each) # f_list : [a_n, a_{n-1}, ..., a_1, a_0]
            fe_w_2i.append(result % prime_field)  

        for each in w_2i :
            result = evaluate_polynomial(fo_list, each) # f_list : [a_n, a_{n-1}, ..., a_1, a_0]
            fo_w_2i.append(result % prime_field)

        end3_comp_time = time.time ()

        comp_time += end3_comp_time - start3_comp_time  

        reg_sheet1.row += [VID, PID, VPID, NVID, listToString(f_w_i), listToString(fe_w_2i), listToString(fo_w_2i), comp_time]
        reg_sheet1.save_as ("STK_Reg_Veh_details.xlsx")

        print ("\n\nTotal Veh Comp time is ", comp_time, "\n===================")