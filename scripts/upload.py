import paramiko
import os, time, requests

def uploadImage(filename = 'bus.jpg'):
    start_time = time.time()
    hostname = '202.120.48.71'
    port = 22000
    username = 'xin99'
    password = 'xin990114'

    local_image_path = "data/data_collection/" + filename
    remote_image_path = "/home/xin99/BloodScan/data_collection/" + filename
    remote_annotated_image_path = "/home/xin99/BloodScan/data_collection/annotated/" + filename
    local_annotated_image_path = "data/data_collection/annotated/" + filename
    remote_script_path = "/home/xin99/BloodScan/tests/yolo_seg/yolo_valid_remote.py"

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port=port, username=username, password=password)
    
    # 上传图片到云端
    sftp_client = ssh_client.open_sftp()
    sftp_client.put(local_image_path, remote_image_path)

    command = f"""
source /home/xin99/anaconda3/etc/profile.d/conda.sh && conda activate bloodScan && python3 {remote_script_path} --input {remote_image_path} --output {remote_annotated_image_path}
"""

    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()
    if output:
        print("Inference Output:")
        print(output)
    if error:
        print("Inference Error:")
        print(error)

    sftp_client.get(remote_annotated_image_path, local_annotated_image_path)
    sftp_client.close()
    ssh_client.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Cloud processing took {elapsed_time:.2f} seconds.")

def uploadImage2Flask(filename = 'bus.jpg'):
    hostname = '202.120.48.71'
    port = 22000
    username = 'xin99'
    password = 'xin990114'

    local_image_path = "data/data_collection/" + filename
    remote_image_path = "/home/xin99/BloodScan/data_collection/" + filename
    remote_annotated_image_path = "/home/xin99/BloodScan/data_collection/annotated/" + filename
    local_annotated_image_path = "data/data_collection/annotated/" + filename
    remote_script_path = "/home/xin99/BloodScan/tests/yolo_seg/yolo_valid_remote.py"
    
    start_time_ = time.time()
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port=port, username=username, password=password)
    
    sftp_client = ssh_client.open_sftp()
    sftp_client.put(local_image_path, remote_image_path)
    # ssh -L 5000:127.0.0.1:5000 xin99@202.120.48.71 -p 22000
    # conda activate bloodScan
    # cd /home/xin99/BloodScan/tests/yolo_seg
    # python yolo_service.py
    # url = "http://192.168.1.116:5000/infer"

    url = "http://localhost:5000/infer"
    payload = {
        "input_image": remote_image_path,
        "output_image": remote_annotated_image_path
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Inference completed successfully.")
        sftp_client.get(remote_annotated_image_path, local_annotated_image_path)
    else:
        print("Inference failed.")
        print(response.json())
    sftp_client.close()
    ssh_client.close()

    end_time_ = time.time()
    elapsed_time_ = end_time_ - start_time_
    print(f"SSH+SFTP+Inference took {elapsed_time_:.2f} seconds.")
    
    return local_annotated_image_path

if __name__ == "__main__":
    uploadImage2Flask()