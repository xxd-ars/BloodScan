# SSH登录服务器后执行
cd ~/BloodScan/dual_yolo
rsync -av --exclude='sample_visualizations' evaluation_results_v2/ evaluation_results_v2_novis/

# 本地Windows传输
scp -P 22000 -r xin99@202.120.48.71:~/BloodScan/dual_yolo/evaluation_results_v2_novis/* "C:/Users/ASUS/Documents/SJTU M2/Graduation Project/BloodScan/dual_yolo/evaluation_results_v2_novis/"

ssh -p 22000 xin99@202.120.48.71 "rm -rf ~/BloodScan/dual_yolo/evaluation_results_v2_novis"