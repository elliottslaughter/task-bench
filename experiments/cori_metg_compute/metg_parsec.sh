#!/bin/bash
#SBATCH --account=m2294
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

module unload PrgEnv-intel
module load PrgEnv-gnu
#module load openmpi

cores=$(( $(echo $SLURM_JOB_CPUS_PER_NODE | cut -d'(' -f 1) / 2 ))

function launch {
    srun -n $1 -N $1 --cpus-per-task=$cores --cpu_bind none ../../parsec/main "${@:2}" -field 2 -c $cores -p 1 -S $cores
}

function repeat {
    local -n result=$1
    local n=$2
    result=()
    for i in $(seq 1 $n); do
        result+=("${@:3}")
        if (( i < n )); then
            result+=("-and")
        fi
    done
}

function sweep {
    for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
        for rep in 0 1 2 3 4; do
            if [[ $rep -le $s ]]; then
                local args
                repeat args ${NGRAPHS:-1} -kernel compute_bound -iter $(( 1 << (26-s) )) -type $3 -radix ${RADIX:-5} -steps ${STEPS:-1000} -width $(( $2 * cores ))
                $1 $2 "${args[@]}"
            fi
        done
    done
}

for n in $SLURM_JOB_NUM_NODES; do
    for t in ${PATTERN:-stencil_1d}; do
        sweep launch $n $t > parsec_type_${t}_nodes_${n}.log
    done
done
