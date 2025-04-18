import ReadIMU as ReadIMU 
import time  
# from DNN_torch import DNN  
from DNN_torch_ram import DNNRam, LSTMNetwork  
import torch 
import datetime  
import numpy as np   
import csv                          


def load_nn(
        saved_policy_path = "current_exo.pt",  
        nn_type='nn',
        kp=None, 
        kd=None   
    ):   
    hip_nn = None  
    if nn_type == 'lstm':   
        hip_nn = LSTMNetwork(n_input=4, n_layer_1=256, num_layers=2, n_output=2, kp=kp, kd=kd)        
        hip_nn.load_saved_policy(torch.load(saved_policy_path, map_location=torch.device('cpu')))     
    else: 
        hip_nn = DNNRam(18, 128, 64, 2, saved_policy_path=saved_policy_path, kp=kp, kd=kd)     
    return hip_nn    


ComPort = '/dev/serial0'     
imu = ReadIMU.READIMU(ComPort)      

start = time.time()   

kp = 50
kd = 0.1 * np.sqrt(kp)       

# dnn = DNN(18, 128, 64, 2)  # depends on training network  
dnn = load_nn(
    # saved_policy_path = "./nn_para/lstm/current_exo.pt",  
    # nn_type           = 'lstm'
    saved_policy_path = "./nn_para/mlp/current_exo.pt",   
    nn_type='nn', 
    kp = kp, 
    kd = kd 
) 

now = 0  
t_pr1 = 0   
t_pr2 = 0   
t_pr3 = 0   
pk = 0   
counter = 0     
L_Cmd = 0   
R_Cmd = 0   
Cmd_MIN = -15.0 
Cmd_MAX = 15.0 

L_Ctl = 1   
R_Ctl = 1     
Cmd_scale = 1     

# command: 1.5 for running, 2 for climbing  
kcontrol = 1.0         
output = np.array([])    

date = time.localtime(time.time())  
date_year = date.tm_year  
date_month = date.tm_mon  
date_day = date.tm_mday  
date_hour = date.tm_hour  
date_minute = date.tm_min  
date_second = date.tm_sec   

root_path = "./data/Jimmy/s0x75-"  # "s1x25, s1x75, sx20"
# Create filename with format {Year}{Month}{Day}-{Hour}{Minute}{Second}.csv
csv_filename = root_path + f"{date_year:04}{date_month:02}{date_day:02}-{date_hour:02}{date_minute:02}{date_second:02}.csv"


with open(csv_filename, 'a', newline='') as csvfile: 
    fieldnames = ['L_IMU_Ang', 'R_IMU_Ang', 'L_IMU_Vel', 'R_IMU_Vel', 'L_Cmd', 'R_Cmd', 'Peak', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 

    # Write the header only if the file is empty
    csvfile.seek(0, 2)   
    if csvfile.tell() == 0:    
        writer.writeheader()    

    while True:
        now = (time.time() - start)  
        
        if now > 360:   
            break  
        
        imu.read()    
        imu.decode()    
        # print("count :", counter)    

        counter = counter + 1 
        
        L_IMU_angle = imu.XIMUL 
        R_IMU_angle = imu.XIMUR 
        L_IMU_vel = imu.XVIMUL 
        R_IMU_vel = imu.XVIMUR 
        
        t_pr1 = now 
        
        # print(f"Time when running NN = {now:^8.3f}")  
        dnn.generate_assistance(L_IMU_angle, R_IMU_angle, L_IMU_vel, R_IMU_vel, kp, kd)  

        L_Cmd = L_Ctl * dnn.hip_torque_L * kcontrol
        R_Cmd = R_Ctl * dnn.hip_torque_R * kcontrol   
        
        L_Cmd = np.clip(L_Cmd, Cmd_MIN, Cmd_MAX)    
        R_Cmd = np.clip(R_Cmd, Cmd_MIN, Cmd_MAX)     
        
        # if (L_Cmd > pk or R_Cmd > pk):
        #     if (R_Cmd > L_Cmd):
        #         pk = R_Cmd
        #     if (L_Cmd > R_Cmd):
        #         pk = L_Cmd   

        B1_int16 = int(imu.ToUint(L_Cmd/Cmd_scale, -20, 20, 16))      
        B2_int16 = int(imu.ToUint(R_Cmd/Cmd_scale, -20, 20, 16))      

        b1 = (B1_int16 >> 8 & 0x00ff)
        b2 = (B1_int16 & 0x00FF)  
        b3 = (B2_int16 >> 8 & 0x00ff)  
        b4 = (B2_int16 & 0x00FF) 

        imu.send(b1, b2, b3, b4)   

        data = {
            'Time': now, 
            'L_IMU_Ang': L_IMU_angle,
            'R_IMU_Ang': R_IMU_angle,
            'L_IMU_Vel': L_IMU_vel,
            'R_IMU_Vel': R_IMU_vel,
            'L_Cmd': L_Cmd/Cmd_scale,  
            'R_Cmd': R_Cmd/Cmd_scale,  
            'Peak': pk
        }   
        writer.writerow(data) 
        csvfile.flush()  # Ensure data is written to file   
        print(f"| now: {now:^8.3f} | L_IMU_Ang: {L_IMU_angle:^8.3f} | R_IMU_Ang: {R_IMU_angle:^8.3f} | L_IMU_Vel: {L_IMU_vel:^8.3f} | R_IMU_Vel: {R_IMU_vel:^8.3f} | L_Cmd: {L_Cmd/Cmd_scale:^8.3f} | R_Cmd: {R_Cmd/Cmd_scale:^8.3f} | Peak: {pk/Cmd_scale:^8.3f} |")