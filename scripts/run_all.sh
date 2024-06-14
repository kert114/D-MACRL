#!/bin/bash

# Run run_exact_var.sh
nohup ./run_exact_var.sh &
wait

# Run run_pod.sh
nohup ./run_pod.sh &
wait

# # Run run_edge_var.sh
# nohup ./run_edge_var.sh &
# wait

# # Run run_dec_var.sh
# nohup ./run_dec_var.sh &
# wait

# # Run run_fed_sim.sh
# nohup ./run_fed_sim.sh &
# wait

# # Run run_reg_sim.sh
# nohup ./run_reg_sim.sh &
# wait

echo "All jobs have been completed successfully."