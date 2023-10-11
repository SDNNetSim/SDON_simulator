#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <start_date>"
    echo "Example: $0 '7 days ago'"
    exit 1
fi

START_DATE=$(date -d "$1" "+%Y-%m-%d")
GROUP_USERS=("ryan_mccann_student_uml_edu" "arash_rezaee_student_uml_edu" "shamsunnahar_edib_student_uml_edu" "kenneth_watts_student_uml_edu")

print_user_resources() {
    local user=$1
    local is_current=$2
    local total_cpus=0
    local total_mem_mb=0
    local total_nodes=0
    local job_info=""

    if [ "$is_current" = true ]; then
	job_info=$(squeue -u $user --noheader --format="%i %C %m %D")
    else
	job_info=$(sacct -u $user --noheader -o JobID,AllocCPUS,ReqMem,AllocNodes -S "$START_DATE")
    fi

    while IFS= read -r line; do
        local jobid=$(echo $line | awk '{print $1}')
        local cpus=$(echo $line | awk '{print $2}')
        local mem_spec=$(echo $line | awk '{print $3}')
        local nodes=$(echo $line | awk '{print $4}')

        local mem_value=${mem_spec%[cCnN]*}
        local mem_type=${mem_spec: -1}
        
        if [[ $mem_value =~ "G" ]]; then
            mem_value=$(echo $mem_value | tr -d 'G')
            mem_value=$((mem_value * 1024))
        else
            mem_value=$(echo $mem_value | tr -d 'M')
        fi

        if [ "$mem_type" == "c" ] || [ "$mem_type" == "C" ]; then
            total_mem_mb=$((total_mem_mb + mem_value * cpus))
        else
            total_mem_mb=$((total_mem_mb + mem_value))
        fi

        total_cpus=$((total_cpus + cpus))
        total_nodes=$((total_nodes + nodes))
    done <<< "$job_info"

    local total_mem_gb=$(echo "scale=2; $total_mem_mb / 1024" | bc)

    echo "User $user is using $total_cpus CPUs, $total_mem_gb GB (or $total_mem_mb MB) of memory, and $total_nodes nodes."
}

echo "Resource usage from $START_DATE to now:"
for group_user in "${GROUP_USERS[@]}"; do
    print_user_resources $group_user false
done

echo ""
echo "Resource usage for currently running jobs:"
for group_user in "${GROUP_USERS[@]}"; do
    print_user_resources $group_user true
done
