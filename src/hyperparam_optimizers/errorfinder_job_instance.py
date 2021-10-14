import os
from posixpath import commonpath
import sys
import subprocess
import random

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HyperParameterOptimizer import SearchJobInstance


class ErrorfinderJobInstance(SearchJobInstance):
    def __init__(self, id):
        super().__init__(id)
        with open('job_details.txt', 'r') as file:
            self.job_details = file.read()

    def launch(self,
               batch_size,
               lr,
               num_neurons_pos,
               num_hidden_layers_pos,
               num_neurons_neg,
               num_hidden_layers_neg ,
               lagrange_missclassification,
               lagrange_pnorm,
               lagrange_smalltozero,
               pnorm_order,
               plot=False):
        self.passed_args = locals()

        shell_file = self.job_details + "#$ -o signatures-net/tmp/Cluster/errorfinder_%s.out"%str(self.id) + '\n' + '\n'
        args = "--model_id=" + str(self.id)
        args += " --batch_size=" + str(batch_size)
        args += " --lr=" + str(lr)
        args += " --num_neurons_pos=" + str(num_neurons_pos)
        args += " --num_hidden_layers_pos=" + str(num_hidden_layers_pos)
        args += " --num_neurons_neg=" + str(num_neurons_neg)
        args += " --num_hidden_layers_neg=" + str(num_hidden_layers_neg)
        args += " --lagrange_missclassification=" + str(lagrange_missclassification)
        args += " --lagrange_pnorm=" + str(lagrange_pnorm)
        args += " --lagrange_smalltozero=" + str(lagrange_smalltozero) 
        args += " --pnorm_order=" + str(pnorm_order) 
        shell_file += "cd signatures-net/src/ ; conda activate sigs_env ; python train_errorfinder.py " + args

        command = "echo '" + shell_file + "' | ssh cserranocolome@ant-login.linux.crg.es -T 'cat  > signatures-net/tmp/error_" + str(self.id) + ".sh'" 
        command2 = "ssh cserranocolome@ant-login.linux.crg.es 'qsub -N errorfinder_" + str(self.id) + " signatures-net/tmp/error_" + str(self.id) + ".sh'" 
        self.process = subprocess.Popen(command, shell=True)
        time.sleep(1)
        self.process = subprocess.Popen(command2, shell=True)
        print("Launch DONE!")

    def get_result(self):
        import shlex
        command = "ssh cserranocolome@ant-login.linux.crg.es "
        command += "cat signatures-net/tmp/score_%s.txt"%self.id
        output = float(subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")[0])
        return output

    def done(self):
        import shlex
        command = "ssh cserranocolome@ant-login.linux.crg.es "
        command += "ls signatures-net/tmp/"
        output = subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")
        is_done = "score_%s.txt"%self.id in output
        return is_done

    def kill(self):
        # os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        pass

    def end(self):
        pass


if __name__ == '__main__':
    errorfinder_job_instance = ErrorfinderJobInstance(1)
    errorfinder_job_instance.launch(500, 1e-4, 300, 3, 200, 2, 7e-3, 1e4, 1.0, 5.0, plot=True)
