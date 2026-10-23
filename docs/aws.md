# Running on AWS

Use this once the course AWS credits ($150 per student) are approved. The scripts are the same as for a laptop; follow the main README from step 1 on the instance. Set `export NUPLAN_HOME=/data/nuplan` if you mount a separate data volume.

## Instance

| | Recommendation | Why |
|---|---|---|
| Instance | **g5.8xlarge** (1× A10G 24 GB, 32 vCPU, 128 GB RAM) | nuPlan simulation is **CPU-bound**: one ray worker runs per core. The planner model is small, so one GPU is enough. Use g5.16xlarge (64 vCPU) to finish runs about twice as fast. |
| AMI | *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)* | It ships with the NVIDIA driver. `bootstrap.sh` installs everything else. |
| Storage | gp3 EBS, **≥ 500 GB** (maps + val + test + a slice of train), mounted at `/data` | Check the archive sizes on the nuPlan download page before you settle on a size. |
| Region | us-east-1 or us-east-2 | g5 availability is good there. nuPlan downloads come from S3 in the US. |

Cost hygiene: **stop the instance when idle**. The EBS volume keeps the data. Set an AWS Budget alert.
If you share one box, give each teammate their own Linux user and keep a single shared copy of `/data/nuplan`.

Attaching a separate data volume (skip this if you just made the root volume large):
```bash
lsblk                                   # find the new device, e.g. nvme1n1
sudo mkfs -t ext4 /dev/nvme1n1 && sudo mkdir -p /data && sudo mount /dev/nvme1n1 /data
echo "/dev/nvme1n1 /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```


## Budget

- **Stop the instance whenever nothing is running.** One g5.8xlarge left on over a weekend costs about $120.
- Set AWS Budget alerts at $100 and $300.
- Use spot instances for short or resumable jobs, and a small instance (g5.xlarge) for setup and debugging.
