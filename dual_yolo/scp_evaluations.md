# SSH登录服务器后执行
cd ~/BloodScan/dual_yolo
rsync -av --exclude='sample_visualizations' evaluation_results_v4/ evaluation_results_v4_novis/
rsync -av --exclude='visualizations' evaluation_results_v5/ evaluation_results_v5_novis/

# 本地Windows传输
scp -P 22000 -r xin99@202.120.48.71:~/BloodScan/dual_yolo/evaluation_results_v4_novis/* "C:/Users/ASUS/Documents/SJTU M2/Graduation Project/BloodScan/dual_yolo/evaluation_results_v4_novis/"

ssh -p 22000 xin99@202.120.48.71 "rm -rf ~/BloodScan/dual_yolo/evaluation_results_v4_novis"