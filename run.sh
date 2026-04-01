# nsys profile --trace=cuda,nvtx --output=results/host_to_device_ce_ce --force-overwrite true ./nvbandwidth -t host_to_device_ce_ce -v > results/host_to_device_ce_ce.txt
# nsys profile --trace=cuda,nvtx --output=results/host_to_device_ce_sm --force-overwrite true ./nvbandwidth -t host_to_device_ce_sm -v > results/host_to_device_ce_sm.txt
# nsys profile --trace=cuda,nvtx --output=results/device_to_host_ce_ce --force-overwrite true ./nvbandwidth -t device_to_host_ce_ce -v > results/device_to_host_ce_ce.txt
# nsys profile --trace=cuda,nvtx --output=results/device_to_host_ce_sm --force-overwrite true ./nvbandwidth -t device_to_host_ce_sm -v > results/device_to_host_ce_sm.txt

nsys profile --trace=cuda,nvtx --output=results/device_to_device_read_ce_ce  --force-overwrite true ./nvbandwidth -t device_to_device_read_ce_ce  --cooldown 24 -v > results/device_to_device_read_ce_ce.txt
nsys profile --trace=cuda,nvtx --output=results/device_to_device_read_ce_sm  --force-overwrite true ./nvbandwidth -t device_to_device_read_ce_sm  --cooldown 8  -v  > results/device_to_device_read_ce_sm.txt
nsys profile --trace=cuda,nvtx --output=results/device_to_device_write_ce_ce --force-overwrite true ./nvbandwidth -t device_to_device_write_ce_ce --cooldown 8 -v > results/device_to_device_write_ce_ce.txt
nsys profile --trace=cuda,nvtx --output=results/device_to_device_write_ce_sm --force-overwrite true ./nvbandwidth -t device_to_device_write_ce_sm --cooldown 12 -v > results/device_to_device_write_ce_sm.txt