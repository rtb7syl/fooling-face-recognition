import os
import subprocess


print("Current working directory: {0}".format(os.getcwd()))

home = os.getcwd()+"/"
os.chdir(home)

job_name ="adv_mask_train"
mail_id="ritabrata.sanyal@rwth-aachen.de"

command = "sbatch -J " + job_name + " -o " + home+"cluster_out/" + job_name + "_out_test.txt -e " + home+"cluster_err/" + job_name + "_err_test.txt "
command += "--account lect0083 -t 6:00:00 --mem 50G -c 10 --gres gpu:1 --partition c16g " + home+"testscript.sh "+"--mail-type begin --mail-type end --mail-type fail --mail-user "+mail_id

os.system(command + " " + job_name)
print(command + " " + job_name)



