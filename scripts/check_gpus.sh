for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    ssh -q "$node" "
      host=\$(hostname)
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
      | awk -v host=\"\$host\" 'BEGIN { OFS=\",\" } { print host, \$0 }'
    " 2>/dev/null
  done
