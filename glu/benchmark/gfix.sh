#!/bin/bash

# show current time
start_time=$(date +%s)

mkdir -p ../conf/S24T24_cg
for n_conf in $(seq 0 49); do
    echo " "
    echo "Processing configuration ${n_conf}"

    gauge_file=../conf/S24T24/wilson_b6.${n_conf}
    gfixed_file=../conf/S24T24_cg/wilson_b6.cg.1e-14.${n_conf}

    config_start_time=$(date +%s)
    echo " "
    echo "Start time for config ${n_conf}: $(date)"

    ./GLU -i input_S24T24.txt -c ${gauge_file} -o ${gfixed_file}

    config_end_time=$(date +%s)
    config_elapsed_time=$((config_end_time - config_start_time))

    echo " "
    echo "Time for config ${n_conf}: $((config_elapsed_time / 3600)) hours $(((config_elapsed_time % 3600) / 60)) minutes $((config_elapsed_time % 60)) seconds"
done

# calculate total time
end_time=$(date +%s)
total_time=$(echo "$end_time - $start_time" | bc)
echo " "
echo "Total time: $total_time seconds"
