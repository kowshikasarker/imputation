import os
import sys

orig_stdout = sys.stdout
orig_stderr = sys.stderr
log_file = open('/shared/nas/data/m1/ksarker2/Imputation/Code/run-reaction-coparticipation-based-imputation.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

test_dir = '/shared/nas/data/m1/ksarker2/Imputation/Data/Metabolome/Datasets/Test_Data'
results_dir = '/shared/nas/data/m1/ksarker2/Imputation/Results/Test_Data'

datasets = os.listdir(test_dir)

script = '/shared/nas/data/m1/ksarker2/Imputation/Code/reaction-coparticipation-based-imputation.py'

for dataset in datasets:
    test_cases = [dir_name for dir_name in os.listdir(test_dir + '/' + dataset) if not os.path.isfile(test_dir + '/' + dataset + '/' + dir_name)]
    
    base_command = 'python3 ' + script + ' -g /shared/nas/data/m1/ksarker2/Imputation/Data/Knowledge/Human-GEM-1.18.0/Human-GEM.xlsx -m /shared/nas/data/m1/ksarker2/Imputation/Data/Knowledge/Human-GEM-1.18.0/metabolites.tsv' 
    print('dataset', dataset)
    for test_case in test_cases:
        print('test_case', test_case)
        in_path = test_dir + '/' + dataset + '/' + test_case + '/' + dataset + '_' + test_case + '.tsv'
        out_dir = results_dir + '/' + dataset + '/' + test_case
        print('in_path', in_path)
        print('out_dir', out_dir)
        command = base_command + ' -i ' + in_path + ' -o ' + out_dir
        os.system(command)
        break
        
sys.stdout = orig_stdout 
sys.stderr = orig_stderr 
log_file.close()
        
        
    
    
    
    
     