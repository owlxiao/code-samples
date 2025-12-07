import psutil
import os 

# calculate the maximum allowed NUM_JOBS based on cores
max_num_jobs_cores = max(1, os.cpu_count() // 2)

# calculate the maximum allowed NUM_JOBS based on free memory
free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

# pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))

print(max_jobs)