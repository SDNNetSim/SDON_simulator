#!/bin/bash

#SBATCH -p cpu-preempt
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/SDN_Simulator/

config_one="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_one', 'learn_rate': 0.1, 'discount': 0.9}"
config_two="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_one', 'learn_rate': 0.9, 'discount': 0.1}"
config_three="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_one', 'learn_rate': 0.5, 'discount': 0.5}"

config_four="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_two', 'learn_rate': 0.1, 'discount': 0.9}"
config_five="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_two', 'learn_rate': 0.9, 'discount': 0.1}"
config_six="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_two', 'learn_rate': 0.5, 'discount': 0.5}"

config_seven="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_three', 'learn_rate': 0.1, 'discount': 0.9}"
config_eight="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_three', 'learn_rate': 0.9, 'discount': 0.1}"
config_nine="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_three', 'learn_rate': 0.5, 'discount': 0.5}"

config_ten="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_four', 'learn_rate': 0.1, 'discount': 0.9}"
config_eleven="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_four', 'learn_rate': 0.9, 'discount': 0.1}"
config_twelve="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_four', 'learn_rate': 0.5, 'discount': 0.5}"

config_thirteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_five', 'learn_rate': 0.1, 'discount': 0.9}"
config_fourteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_five', 'learn_rate': 0.5, 'discount': 0.5}"
config_fifteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_five', 'learn_rate': 0.9, 'discount': 0.1}"

config_sixteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_six', 'learn_rate': 0.1, 'discount': 0.9}"
config_seventeen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_six', 'learn_rate': 0.5, 'discount': 0.5}"
config_eighteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_six', 'learn_rate': 0.9, 'discount': 0.1}"

config_nineteen="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_seven', 'learn_rate': 0.1, 'discount': 0.9}"
config_twenty="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_seven', 'learn_rate': 0.5, 'discount': 0.5}"
config_twentyone="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_seven', 'learn_rate': 0.9, 'discount': 0.1}"

config_twentytwo="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_eight', 'learn_rate': 0.1, 'discount': 0.9}"
config_twentythree="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_eight', 'learn_rate': 0.5, 'discount': 0.5}"
config_twentyfour="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_eight', 'learn_rate': 0.9, 'discount': 0.1}"

config_twentyfive="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_nine', 'learn_rate': 0.1, 'discount': 0.9}"
config_twentysix="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_nine', 'learn_rate': 0.5, 'discount': 0.5}"
config_twentyseven="{'is_training': 'True', 'table_path': 'None', 'epsilon': 0.05, 'epsilon_target': 0.01, 'policy': 'policy_nine', 'learn_rate': 0.9, 'discount': 0.1}"

ai_array=("$config_one" "$config_two" "$config_three" "$config_four" "$config_five" "$config_six" "$config_seven" "$config_eight" "$config_nine" "$config_ten" "$config_eleven" "$config_twelve" "$config_thirteen" "$config_fourteen" "$config_fifteen" "$config_sixteen" "$config_seventeen" "$config_eighteen" "$config_nineteen" "$config_twenty" "$config_twentyone" "$config_twentytwo" "$config_twentythree" "$config_twentyfour" "$config_twentyfive" "$config_twentysix" "$config_twentyseven")

python run_sim.py --ai_arguments "${ai_array[$SLURM_ARRAY_TASK_ID]}"
