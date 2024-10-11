# Introduction  
Source code for the paper entitled "Advanced Hypergame-Theoretic Defensive APT Deception for Securing Data Communication in IIoT Control Systems" submitted to IEEE Transactions on Information Forensics and Security for possible publication.
# Requirements  
Linux  
Python 3.6+  
gym  
TensorFlow 1.x
# Environment  
Comparative metrics： Mean Time Between Failures,False Alarm Rate, Detection Rate
# One defender versus multiple APT attackers(Figs. 5,6)  
Comparison Algorithms : HG-CG-D3QN,TG-CG-D3QN,HG-ML,TG-ML  
- HG-CG-D3QN,TH-CG-D3QN:   
python ML_D3QN_train_strat_sep.py --algo d3qn --strategy separated  
- HG-ML,TH-ML:  
python ML_training.py --max_depth 5 --criterion gini --dataset dataset.csv
# Learning rate (Fig. 4)  
Modify the learning_rate pass parameter of the D3QN_Network class in the Foureye display.py file  
- HG-CG-D3QN,TH-CG-D3QN:  
 python ML_D3QN_train_strat_sep.py --algo d3qn --strategy separated  




 



 












