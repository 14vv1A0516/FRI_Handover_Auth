from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.node import OVSKernelAP, UserAP
from mn_wifi.link import wmediumd, _4address
from mn_wifi.wmediumdConnector import interference
from mininet.log import setLogLevel, info
from mininet.term import makeTerm 
from mininet.link import TCLink 
from mininet.node import OVSKernelSwitch, RemoteController
import os, time

'''
os.system ("pip install pyexcel ") # --break-system-packages
os.system ("pip install  ecdsa  ")
os.system ("pip install  eciespy  ")
os.system ("pip install  openpyxl==3.0.10   ")
os.system ("pip install  pyexcel==0.6.7  ") 
os.system ("pip install  pyexcel-xlsx==0.6.0  ") 
'''

def topology():
    
    #net = Mininet_wifi (link=TCLink, accessPoint= UserAP) controller= RemoteController,
    #net = Mininet_wifi (controller= RemoteController, accessPoint=OVSKernelAP, link=wmediumd, wmediumd_mode=interference) #accessPoint= UserAP) # TCLink
    net = Mininet_wifi (allAutoAssociation=True, controller= RemoteController, accessPoint=OVSKernelAP, link=wmediumd, wmediumd_mode=interference) #accessPoint= UserAP) # TCLink

    info ("Creating nodes ---- \n")

    sta1 = net.addStation ("sta1", wlans=1, mac="00:00:00:00:00:01", ip="10.0.0.1/8", min_v=1.0, max_v=5.0, range=4) # position="45,105,0", 192.168.0.3/24  min_v=5.0, max_v=10.0,
    #sta1 = net.addStation ("sta1", wlans=1, mac="00:00:00:00:00:01", ip="10.0.0.1/8", max_x=10, max_y=10, min_v=5.0, max_v=10.1, range=4) # position="45,105,0", 192.168.0.3/24
     
    sta2 = net.addStation('sta2', wlans=1, mac='00:00:00:00:00:02', ip='10.0.0.2/8', position='10,30,0', min_v=1.0, max_v=5.0, range=5)
    
    sta3 = net.addStation('sta3', mac='00:00:00:00:00:03', ip='10.0.0.3/8', position='10,30,0', min_v=1.0, max_v=5.0, range=5)
    
    sta4 = net.addStation('sta4', mac='00:00:00:00:00:04', ip='10.0.0.4/8', position='10,60,0', min_v=1.0, max_v=5.0, range=5)
    
    sta5 = net.addStation('sta5', mac='00:00:00:00:00:05', ip='10.0.0.5/8', position='90,60,0',  min_v=1.0, max_v=5.0, range=5)
    
    sta6 = net.addStation('sta6', mac='00:00:00:00:00:06', ip='10.0.0.6/8', position='90,60,0',  min_v=1.0, max_v=5.0, range=5)
    '''
    sta7 = net.addStation('sta7', mac='00:00:00:00:00:07', ip='10.0.0.7/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta8 = net.addStation('sta8', mac='00:00:00:00:00:08', ip='10.0.0.8/8', position='90,60,0', min_v=1.0, max_v=5.0, range=5)
    sta9 = net.addStation('sta9', mac='00:00:00:00:00:09', ip='10.0.0.9/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta10 = net.addStation('sta10', mac='00:00:00:00:00:10', ip='10.0.0.10/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta11 = net.addStation('sta11', mac='00:00:00:00:00:11', ip='10.0.0.11/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta12 = net.addStation('sta12', mac='00:00:00:00:00:12', ip='10.0.0.12/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta13 = net.addStation('sta13', mac='00:00:00:00:00:13', ip='10.0.0.13/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta14 = net.addStation('sta14', mac='00:00:00:00:00:14', ip='10.0.0.14/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta15 = net.addStation('sta15', mac='00:00:00:00:00:15', ip='10.0.0.15/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta16 = net.addStation('sta16', mac='00:00:00:00:00:16', ip='10.0.0.16/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta17 = net.addStation('sta17', mac='00:00:00:00:00:17', ip='10.0.0.17/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta18 = net.addStation('sta18', mac='00:00:00:00:00:18', ip='10.0.0.18/8', position='90,60,0', min_v=1.0, max_v=5.0,   range=5)
    sta19 = net.addStation('sta19', mac='00:00:00:00:00:19', ip='10.0.0.19/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta20 = net.addStation('sta20', mac='00:00:00:00:00:20', ip='10.0.0.20/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta21 = net.addStation('sta21', mac='00:00:00:00:00:21', ip='10.0.0.21/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta22 = net.addStation('sta22', mac='00:00:00:00:00:22', ip='10.0.0.22/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta23 = net.addStation('sta23', mac='00:00:00:00:00:23', ip='10.0.0.23/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta24 = net.addStation('sta24', mac='00:00:00:00:00:24', ip='10.0.0.24/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta25 = net.addStation('sta25', mac='00:00:00:00:00:25', ip='10.0.0.25/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta26 = net.addStation('sta26', mac='00:00:00:00:00:26', ip='10.0.0.26/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    sta27 = net.addStation('sta27', mac='00:00:00:00:00:27', ip='10.0.0.27/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta28 = net.addStation('sta28', mac='00:00:00:00:00:28', ip='10.0.0.28/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta29 = net.addStation('sta29', mac='00:00:00:00:00:29', ip='10.0.0.29/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    sta30 = net.addStation('sta30', mac='00:00:00:00:00:30', ip='10.0.0.30/8', position='90,60,0', min_v=1.0, max_v=5.0,  range=5)
    
    '''
    ap1 = net.addAccessPoint ('RSU1', ssid='SSID_ap1', dpid='1', cls=OVSKernelAP,  mac="00:00:00:00:00:70", mode='g', failMode='standalone', range= '30', channel=4, position='40,40,0', datapath='user') # cls=OVSKernelAP, inNamespace=False,
    ap2 = net.addAccessPoint ('RSU2', ssid='SSID_ap2', dpid='2',  cls=OVSKernelAP, mac="00:00:00:00:00:80", mode='g', failMode='standalone', range= '30', channel=4, position='90,40,0', datapath='user')
    ap3 = net.addAccessPoint ('RSU3', ssid='SSID_ap3', dpid='3', color='r', cls=OVSKernelAP, mac="00:00:00:00:00:90", mode='g', failMode='standalone', range= '30', channel=4, position='150,40,0', datapath='user')
    
    #c1 = net.addController('C1', controller=RemoteController, ip='10.13.1.178', port=6633)
    #c2 = net.addController('C2', controller=RemoteController, ip='10.13.1.224', port=6644)

    net.setPropagationModel (model="logDistance", exp=4.3)

    info ("**** Configuring Wifi Nodes \n")
    net.configureWifiNodes()
    #net.AssociationControl('ssf')

    info ("**** Associating and Creating links \n") 
    
    net.plotGraph (max_x=200, max_y=80)
    net.startMobility (time=0, seed=1, model='logDistance', ac_method='ssf') # can also parameter repetitions=2 - to simulate twice

    net.mobility (sta1, 'start', time=10, position="2,40,0") # 150,105,0 # 1-5 m/s
    net.mobility (sta1, 'stop', time=70, position="180,40,0") # 10,105,0    
    
    net.mobility (sta2, 'start', time=20, position="2,40,0") # 150,105,0
    net.mobility (sta2, 'stop', time=80, position="180,40,0") # 10,105,0
    
    net.mobility (sta3, 'start', time=30, position="2,40,0") # 150,105,0
    net.mobility (sta3, 'stop', time=90, position="180,40,0") # 10,105,0
    
    net.mobility (sta4, 'start', time=40, position="2,40,0") # 150,105,0
    net.mobility (sta4, 'stop', time=100, position="180,40,0") # 10,105,0
    
    net.mobility (sta5, 'start', time=50, position="2,40,0") # 150,105,0
    net.mobility (sta5, 'stop', time=110, position="180,40,0") # 10,105,0
    
    net.mobility (sta6, 'start', time=55, position="2,40,0") # 150,105,0
    net.mobility (sta6, 'stop', time=115, position="180,40,0") # 10,105,0
    '''
    net.mobility (sta7, 'start', time=60, position="2,40,0") # 150,105,0
    net.mobility (sta7, 'stop', time=120, position="180,40,0") # 10,105,0
    
    net.mobility (sta8, 'start', time=70, position="2,40,0") # 150,105,0
    net.mobility (sta8, 'stop', time=130, position="180,40,0") # 10,105,0

    net.mobility (sta9, 'start', time=80, position="2,40,0") # 150,105,0
    net.mobility (sta9, 'stop', time=140, position="180,40,0") # 10,105,0

    net.mobility (sta10, 'start', time=90, position="2,40,0") # 150,105,0
    net.mobility (sta10, 'stop', time=150, position="180,40,0") # 10,105,0
    
    net.mobility (sta11, 'start', time=100, position="2,40,0") # 150,105,0
    net.mobility (sta11, 'stop', time=160, position="180,40,0") # 10,105,0
    
    net.mobility (sta12, 'start', time=110, position="2,40,0") # 150,105,0
    net.mobility (sta12, 'stop', time=170, position="180,40,0") # 10,105,0

    net.mobility (sta13, 'start', time=115, position="2,40,0") # 150,105,0
    net.mobility (sta13, 'stop', time=175, position="180,40,0") # 10,105,0

    net.mobility (sta14, 'start', time=120, position="2,40,0") # 150,105,0
    net.mobility (sta14, 'stop', time=180, position="180,40,0") # 10,105,0

    net.mobility (sta15, 'start', time=130, position="2,40,0") # 150,105,0
    net.mobility (sta15, 'stop', time=190, position="180,40,0") # 10,105,0
    
    net.mobility (sta16, 'start', time=135, position="2,40,0") # 150,105,0
    net.mobility (sta16, 'stop', time=195, position="180,40,0") # 10,105,0
    
    net.mobility (sta17, 'start', time=145, position="2,40,0") # 150,105,0
    net.mobility (sta17, 'stop', time=205, position="180,40,0") # 10,105,0

    net.mobility (sta18, 'start', time=150, position="2,40,0") # 150,105,0
    net.mobility (sta18, 'stop', time=210, position="180,40,0") # 10,105,0

    net.mobility (sta19, 'start', time=160, position="2,40,0") # 150,105,0
    net.mobility (sta19, 'stop', time=220, position="180,40,0") # 10,105,0

    net.mobility (sta20, 'start', time=170, position="2,40,0") # 150,105,0
    net.mobility (sta20, 'stop', time=230, position="180,40,0") # 10,105,0
    
    net.mobility (sta21, 'start', time=180, position="2,40,0") # 150,105,0
    net.mobility (sta21, 'stop', time=240, position="180,40,0") # 10,105,0
    
    net.mobility (sta22, 'start', time=190, position="2,40,0") # 150,105,0
    net.mobility (sta22, 'stop', time=250, position="180,40,0") # 10,105,0

    net.mobility (sta23, 'start', time=195, position="2,40,0") # 150,105,0
    net.mobility (sta23, 'stop', time=255, position="180,40,0") # 10,105,0

    net.mobility (sta24, 'start', time=200, position="2,40,0") # 150,105,0
    net.mobility (sta24, 'stop', time=260, position="180,40,0") # 10,105,0

    net.mobility (sta25, 'start', time=205, position="2,40,0") # 150,105,0
    net.mobility (sta25, 'stop', time=265, position="180,40,0") # 10,105,0
    
    net.mobility (sta26, 'start', time=215, position="2,40,0") # 150,105,0
    net.mobility (sta26, 'stop', time=275, position="180,40,0") # 10,105,0
    
    net.mobility (sta27, 'start', time=225, position="2,40,0") # 150,105,0
    net.mobility (sta27, 'stop', time=285, position="180,40,0") # 10,105,0
    
    net.mobility (sta28, 'start', time=230, position="2,40,0") # 150,105,0
    net.mobility (sta28, 'stop', time=290, position="180,40,0") # 10,105,0

    net.mobility (sta29, 'start', time=240, position="2,40,0") # 150,105,0
    net.mobility (sta29, 'stop', time=300, position="180,40,0") # 10,105,0

    net.mobility (sta30, 'start', time=250, position="2,40,0") # 150,105,0
    net.mobility (sta30, 'stop', time=310, position="180,40,0") # 10,105,0
    
    '''
    
    net.stopMobility (time=450) 
    #ap1.cmd('dhclient ap1-wlan1 -v')
    ap1.cmd('ifconfig RSU1-wlan1 10.0.0.100/8')
    ap2.cmd('ifconfig RSU2-wlan1 11.0.0.100/8')
    
    info ("**** Starting network \n")
    net.build ()
    
    #ap1.start ([c1])
    #ap2.start ([c2])
    #net.build ()


    makeTerm (ap1, cmd = "bash -c 'python3 Auth_RSU.py ;'") # > rsu1.txt
    #time.sleep(1)
    makeTerm (ap2, cmd = "bash -c 'python3 Hand_RSU.py ;'")
    #makeTerm(sta1)
    # makeTerm(sta6)
    #makeTerm (ap1, cmd =  "bash -c 'xterm;'")
    
    veh_ap1_auth_check = {}
    veh_ap2_hand_check = {}
    
    ip_ct = 0
    while True :
        for sta in net.stations : 
            if str(sta) not in veh_ap1_auth_check and sta.wintfs[0].associatedTo is not None :  
                #print ("Enter if ...")
                veh_ap1_auth_check[str(sta)] = 0
                veh_ap2_hand_check[str(sta)] = 0
                apx = sta.wintfs[0].associatedTo.node
                apx = str(apx)          
                #print ("AP for sta 1 is ", apx)

                if apx == 'RSU1' and veh_ap1_auth_check[str(sta)] == 0 : 
                    
                    if str(sta) == 'sta1': # For Appendix 
                        print ("Enter for sta 1 is ", apx)
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '6JT3S5D' 't1'  > auth_sta1.txt ;'") # > auth_sta1.txt 
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    
                    elif str(sta) == 'sta2':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'PHJVM83' 't2' > auth_sta2.txt ;'") # > auth_sta1.txt
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   
                    
                    elif str(sta) == 'sta3':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'W6DFGCJ' 't3' > auth_sta3.txt ;'") # > auth_sta1.txt
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    elif str(sta) == 'sta4':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'OZAUWHV' 't4' > auth_sta4.txt  ;'") # > auth_sta1.txt
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    elif str(sta) == 'sta5':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'H0SFA52' 't5' > auth_sta5.txt ;'") # > auth_sta1.txt
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta6':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'HUZK0RG' 't6' > auth_sta6.txt  ;'") # > auth_sta1.txt
                        print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    '''
                    elif str(sta) == 'sta7':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'NI8L7OA' 't7' > auth_sta7.txt  ;'") # > auth_sta1.txt
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   
                    
                    elif str(sta) == 'sta8':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'I66S24I' 't8' > auth_sta8.txt  ;'") # > auth_sta1.txt
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta9':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '1L54S4R' 't9' > auth_sta9.txt  ;'") # > auth_sta1.txt

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta10':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '31L1PXM' 't10' > auth_sta10.txt  ;'") # > auth_sta1.txt
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    
                    elif str(sta) == 'sta11':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'EILB01J' 't11' > auth_sta11.txt  ;'") # > auth_sta1.txt
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    

                    elif str(sta) == 'sta12':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '7H338WA' 't12' > auth_sta12.txt  ;'") 
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   

                    elif str(sta) == 'sta13':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'KEDHZ1B' 't13' > auth_sta13.txt  ;'") 

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta14':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'FOIYAJ6' 't14' > auth_sta14.txt  ;'") 

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta15':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'SFLHL1C' 't15' > auth_sta15.txt  ;'") 

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    elif str(sta) == 'sta16':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'QCSR0OR' 't16' > auth_sta16.txt  ;'") 

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    
                    elif str(sta) == 'sta17':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '82I12SA' 't17' > auth_sta17.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   

                    elif str(sta) == 'sta18':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'GYUI6V7' 't18' > auth_sta18.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta19':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'H38JHH0' 't19' > auth_sta19.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta20':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'KKULBJU' 't20' > auth_sta20.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    elif str(sta) == 'sta21':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'LMFYMYU' 't21' > auth_sta21.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    
                    elif str(sta) == 'sta22':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '1RNH2YR' 't22' > auth_sta22.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   

                    elif str(sta) == 'sta23':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'X7HONI7' 't23' > auth_sta23.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta24':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'ATSELAA' 't24' > auth_sta24.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta25':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '47W1ONZ' 't25' > auth_sta25.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 
                    
                    elif str(sta) == 'sta26':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'C1ZMS7F' 't26' > auth_sta26.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1
                    
                    elif str(sta) == 'sta27':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py '5X686V0' 't27' > auth_sta27.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1   

                    elif str(sta) == 'sta28':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'J4VSM0B' 't28' > auth_sta28.txt  ;'")
                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta29':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'JYUH0UP' 't29' > auth_sta29.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1 

                    elif str(sta) == 'sta30':
                        #x = makeTerm (sta, cmd = "bash -c 'python3 Auth_Veh.py 'XVK3I8C' 't30' > auth_sta30.txt  ;'")

                        #print ("-------- Auth done for --- ", str(sta))
                        veh_ap1_auth_check[str(sta)] = 1  
                    '''
            
            
            elif str(sta) in veh_ap1_auth_check and sta.wintfs[0].associatedTo is not None :
                if str(sta.wintfs[0].associatedTo.node) == 'RSU2' and veh_ap2_hand_check[str(sta)] == 0 : 
                        #print ("Into sta and ap2 ...")
                        #print ("Handover ",str(sta)," associated with AP 2) # ", str(sta.wintfs[0].associatedTo.node))     
                        
                        if str(sta) == 'sta1' : # and veh_ap2_hand_check[str(sta)] == 0:  For Appendix
                            
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta1.setIP(IP, intf='sta1-wlan0')
                            print (" sta1 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '6JT3S5D' 't1'  > hand_sta1.txt  ;'") # d
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta2' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta2.setIP(IP, intf='sta2-wlan0')
                            ip_ct = ip_ct + 1
                            print (" sta2 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'PHJVM83' 't2' > hand_sta2.txt  ;'") #d
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta3' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta3.setIP(IP, intf='sta3-wlan0')
                            print (" sta3 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'W6DFGCJ' 't3' > hand_sta3.txt  ;'") #done
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta4' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta4.setIP(IP, intf='sta4-wlan0')
                            print (" sta4 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'OZAUWHV' 't4' > hand_sta4.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta5' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta5.setIP(IP, intf='sta5-wlan0')
                            print (" sta5 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'H0SFA52' 't5' > hand_sta5.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta6' : # and veh_ap2_hand_check[str(sta)] == 0:.
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta6.setIP(IP, intf='sta6-wlan0')
                            print (" sta6 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'HUZK0RG' 't6' > hand_sta6.txt  ;'") # done
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        '''
                        elif str(sta) == 'sta7' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta7.setIP(IP, intf='sta7-wlan0')
                            print (" sta7 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'NI8L7OA' 't7' > hand_sta7.txt  ;'") 
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta8' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta8.setIP(IP, intf='sta8-wlan0')
                            print (" sta8 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'I66S24I' 't8' > hand_sta8.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta9' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta9.setIP(IP, intf='sta9-wlan0')
                            print (" sta9 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '1L54S4R' 't9' > hand_sta9.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta10' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta10.setIP(IP, intf='sta10-wlan0')
                            print (" sta10 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '31L1PXM' 't10' > hand_sta10.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta11' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta11.setIP(IP, intf='sta11-wlan0')
                            print (" sta11 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'EILB01J' 't11' > hand_sta11.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta12' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta12.setIP(IP, intf='sta12-wlan0')
                            print (" sta12 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '7H338WA' 't12' > hand_sta12.txt  ;'") 
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta13' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta13.setIP(IP, intf='sta13-wlan0')
                            print (" sta13 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'KEDHZ1B' 't13' > hand_sta13.txt  ;'") 
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta14' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta14.setIP(IP, intf='sta14-wlan0')
                            print (" sta14 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'FOIYAJ6' 't14' > hand_sta14.txt  ;'") 
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta15' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta15.setIP(IP, intf='sta15-wlan0')
                            print (" sta15 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'SFLHL1C' 't15' > hand_sta15.txt  ;'") 
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta16' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta16.setIP(IP, intf='sta16-wlan0')
                            print (" sta16 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'QCSR0OR' 't16' > hand_sta16.txt  ;'") 
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta17' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta17.setIP(IP, intf='sta17-wlan0')
                            print (" sta17 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '82I12SA' 't17' > hand_sta17.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta18' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta18.setIP(IP, intf='sta18-wlan0')
                            print (" sta18 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'GYUI6V7' 't18' > hand_sta18.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta19' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta19.setIP(IP, intf='sta19-wlan0')
                            print (" sta19 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'H38JHH0' 't19' > hand_sta19.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta20' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta20.setIP(IP, intf='sta20-wlan0')
                            print (" sta20 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'KKULBJU' 't20' > hand_sta20.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta21' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta21.setIP(IP, intf='sta21-wlan0')
                            print (" sta21 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'LMFYMYU' 't21' > hand_sta21.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta22' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta22.setIP(IP, intf='sta22-wlan0')
                            print (" sta22 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '1RNH2YR' 't22' > hand_sta22.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta23' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta23.setIP(IP, intf='sta23-wlan0')
                            print (" sta23 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'X7HONI7' 't23' > hand_sta23.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta24' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta24.setIP(IP, intf='sta24-wlan0')
                            print (" sta24 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'ATSELAA' 't24' > hand_sta24.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta25' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta25.setIP(IP, intf='sta25-wlan0')
                            print (" sta25 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '47W1ONZ' 't25' > hand_sta25.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        
                        elif str(sta) == 'sta26' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta26.setIP(IP, intf='sta26-wlan0')
                            print (" sta26 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'C1ZMS7F' 't26' > hand_sta26.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta27' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta27.setIP(IP, intf='sta27-wlan0')
                            print (" sta27 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py '5X686V0' 't27' > hand_sta27.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta28' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta28.setIP(IP, intf='sta28-wlan0')
                            print (" sta28 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'J4VSM0B' 't28' > hand_sta28.txt  ;'")
                            print ("Handover done for ",str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta29' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta29.setIP(IP, intf='sta29-wlan0')
                            print (" sta29 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'JYUH0UP' 't29' > hand_sta29.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1

                        elif str(sta) == 'sta30' : # and veh_ap2_hand_check[str(sta)] == 0:
                            ip_ct = ip_ct + 1
                            IP = '11.0.0.' + str(ip_ct)
                            sta30.setIP(IP, intf='sta30-wlan0')
                            print (" sta30 IP changed to ", IP)
                            x = makeTerm (sta, cmd = "bash -c 'python3 Hand_Veh.py 'XVK3I8C' 't30' > hand_sta30.txt  ;'")
                            print ("-------- Handover done for --- ", str(sta))
                            veh_ap2_hand_check[str(sta)] = 1
                        '''
            
    info ("**** Running CLI \n")
    CLI(net)

    info ("**** Stopping network \n")
    net.stop ()
    
    
if __name__ == '__main__' :
    setLogLevel ('info')
    topology()




