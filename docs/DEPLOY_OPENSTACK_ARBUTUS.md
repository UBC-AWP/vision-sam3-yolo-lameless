# Deploying the Lameness Detection Platform on OpenStack (Arbutus)

This guide walks you through running the Cow Lameness Detection ML Pipeline on **Arbutus**, the [Digital Research Alliance of Canada](https://alliancecan.ca/) (formerly Compute Canada) OpenStack cloud.

## Prerequisites

- **Alliance/CCDB account** — Same credentials you use for [CCDB](https://ccdb.computecanada.ca/). If you don’t have one, request access through your institution.
- **SSH key pair** — Required for VM login (password login is typically disabled). Generate one if needed:
  ```bash
  ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/arbutus_lameness
  ```
- **Enough quota** — The stack runs 22 services; use a medium-to-large flavor (see below). Check your project quota in the dashboard.

---

## Pre-built images from Docker Hub (recommended)

This deployment guide uses pre-built images from Docker Hub by default. Instead of building all service images on the VM, set your Docker Hub username in `.env` and to prevent rate limiting. All the images currently get built and pushed to [vindictive](https://hub.docker.com/repositories/vindictive). So you might need my account to pull images, otherwise and build and push to ur own docker hub account and login and pull from there.

This mode pulls images from Docker Hub and starts services without local image builds.

---

## Part 1: OpenStack (Arbutus) Setup

### 1.1 Log in to the dashboard

1. Open **https://arbutus.cloud.computecanada.ca/** (or https://arbutus.cloud.alliancecan.ca/).
2. Sign in with your **Alliance username** and **password**.

### 1.2 Create or select a project

- Use an existing OpenStack project or create one: **Identity → Projects → Create Project**.
- Note the project name; you’ll use it when launching instances.

### 1.3 Add your SSH key to OpenStack

1. Go to **Compute → Key Pairs**.
2. Click **Import Key Pair**.
3. **Key pair name**: e.g. `lameness-deploy`.
4. **Public key**: paste the contents of your **public** key (e.g. `~/.ssh/arbutus_lameness.pub`).
   ```bash
   cat ~/.ssh/arbutus_lameness.pub
   ```
5. Click **Import Key Pair**.

### 1.4 Create a security group

Use a **Security Group** (firewall rules for your instance), not a **Server Group** (which is for VM placement/affinity). In the dashboard: **Networks → Security Groups**.

1. Go to **Networks → Security Groups**.
2. Click **Create Security Group**; name it e.g. `lameness-app`.
3. Open the group and click **Add Rule**:
   - **Rule**: SSH (port 22), **Remote**: CIDR, **CIDR**: `0.0.0.0/0` (or your IP for better security).
   - **Rule**: Custom TCP, **Port**: 3000 (frontend), **CIDR**: `0.0.0.0/0` (or restrict).
   - **Rule**: Custom TCP, **Port**: 8000 (backend API), **CIDR**: `0.0.0.0/0` (or restrict).
   - Optionally: 8001 (video-ingestion), 8222 (NATS monitoring) if you need them from outside.
4. Save the rules.

### 1.5 Launch an instance

1. Go to **Compute → Instances**.
2. Click **Launch Instance**.

**Details**

- **Instance Name**: e.g. `lameness-platform`.
- **Count**: 1 (or more if you plan to split services later).

**Source (boot image)**

- **Boot source**: Image.
- Choose an **Ubuntu 24.04** image (or the latest Ubuntu LTS offered). Avoid very old images (e.g. 18.04) for better Docker support.

**Flavor**

- The app runs many containers (PostgreSQL, NATS, Qdrant, 18+ pipelines). Recommended:
  - **Minimum**: 8 vCPUs, 32 GB RAM (e.g. `c8-32gb` or similar, depending on Arbutus flavors).
  - **Comfortable**: 16 vCPUs, 64 GB RAM if available.
- If you need **GPU** for YOLO/SAM3/DINOv3/T-LEAP, choose a GPU flavor if Arbutus provides one and ensure the image has NVIDIA drivers or use a GPU-ready image.

**Networks**

- Attach the **default** project network (or the one your router uses).

**Security groups**

- Attach the security group you created (`lameness-app`).

**Key pair**

- Select the key pair you imported.

3. Click **Launch Instance**. Wait until status is **Active**.

### 1.6 Allocate and associate a floating IP

1. **Compute → Instances**.
2. Find your instance; open the **▼** menu on the right.
3. Click **Associate Floating IP**.
4. Next to **IP Address**, click **+** to **Allocate IP to Project**.
5. **Allocate IP**, then select the new IP and the correct **Port** (your instance).
6. Click **Associate**.

Note: On Arbutus you may have a **limited number of floating IPs** per project (e.g. one). Use it for the VM that will serve the admin UI and API.

### 1.7 Add a volume for persistent data

To keep videos and results across VM rebuilds:

1. **Volumes → Volumes → Create Volume** (e.g. 100–500 GB, name `lameness-data`).
2. After creation, **▼ → Attach Volume** and select your instance; attach as e.g. `/dev/vdb`.
3. On the VM (after first boot), format and mount it, e.g.:
   ```bash
   sudo mkfs.ext4 /dev/vdb
   sudo mkdir -p /mnt/lameness-data
   sudo mount /dev/vdb /mnt/lameness-data
   ```
   Then use `/mnt/lameness-data` as the host path for Docker data (see Part 2).

---

## Part 2: On the VM — Install Docker and run the app

### 2.1 SSH into the instance

Use the **floating IP** you associated:

```bash
ssh -i ~/.ssh/arbutus_lameness ubuntu@<FLOATING_IP>
```

(Replace `ubuntu` with the default user for your image if different, e.g. `centos`.)

### 2.2 Install Docker and Docker Compose

On Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo systemctl status docker --no-pager
```

Log out and back in (or run `newgrp docker`) so `docker` works without `sudo`.

Verify:

```bash
docker --version
docker compose version
```

### 2.3 Clone the repository and configure

```bash
# Clone (use HTTPS or SSH depending on your access)
git clone https://github.com/UBC-AWP/vision-sam3-yolo-lameless
cd vision-sam3-yolo-lameless
```

If you use a **persistent volume** mounted at `/mnt/lameness-data`:

```bash
sudo mkdir -p /mnt/lameness-data/{videos,canonical,processed,training,results,quality_reports}
# Optional: symlink so the app uses the volume
# ln -s /mnt/lameness-data data  # only if you want data entirely on the volume
```

Otherwise the deploy script will create `./data/` under the repo.

Create environment file:

```bash
cp env.example .env
```

**DEPLOY_HOST:** Leave `DEPLOY_HOST=localhost` if you only use the app from the VM (e.g. open http://localhost:3000 in a browser on the VM). Set it to your **floating IP** only if you access the UI from your laptop (http://\<FLOATING_IP\>:3000) and want the printed URLs to match:

```bash
# Optional: only if accessing from outside the VM
sed -i "s/DEPLOY_HOST=localhost/DEPLOY_HOST=<YOUR_FLOATING_IP>/" .env
```

**Use Docker Hub images (no local build):** Add your Docker Hub namespace to `.env`:

```bash
grep -q '^DOCKER_HUB_USER=' .env && \
  sed -i 's/^DOCKER_HUB_USER=.*/DOCKER_HUB_USER=your-dockerhub-username/' .env || \
  echo 'DOCKER_HUB_USER=your-dockerhub-username' >> .env
```

`--skip-build` requires `DOCKER_HUB_USER` and will pull images, then run with `--no-build`.

Before deploy, verify `.env` contains the expected Docker Hub user:

```bash
grep '^DOCKER_HUB_USER=' .env
```

Before running deploy, verify Docker and containerd are using your mounted volume (recommended to avoid `no space left on device` on `/var/lib/containerd/...`):

```bash
docker info | grep "Docker Root Dir"
sudo readlink -f /var/lib/containerd
sudo df -h /var/lib/containerd /mnt/lameness-data /
```

Expected:
- `Docker Root Dir` points to `/mnt/lameness-data/docker-data` (or your chosen volume path)
- `/var/lib/containerd` resolves to `/mnt/lameness-data/containerd-data`

If not configured yet, complete the "Increase disk space for Docker" section first, then continue with deploy.


### 2.4 Deploy the application

**Recommended (Docker Hub pull, no local build):**

```bash
./scripts/deploy.sh --skip-build
```

The script will pull required images from Docker Hub, start PostgreSQL/NATS/Qdrant, initialize the database and Qdrant, then start all services.

**Optional (build on VM):**

```bash
./scripts/deploy.sh
```

Other options:

- Clean start (remove volumes and re-initialize): `./scripts/deploy.sh --clean` or `./scripts/deploy.sh --clean --skip-build`
- Restart and re-pull images from Docker Hub: `./scripts/deploy.sh --skip-build`

The script will:

- Start PostgreSQL, NATS, and Qdrant
- Initialize the database and Qdrant collections
- Pull images (or build if not using `--skip-build`) and start all 22 services

### 2.5 Verify and access

- **Admin UI**: http://\<FLOATING_IP\>:3000  
- **Backend API**: http://\<FLOATING_IP\>:8000  
- **API docs**: http://\<FLOATING_IP\>:8000/docs  

Default admin login: `admin@example.com` / `adminpass123` (change in production).

Useful commands on the VM:

```bash
docker compose ps
docker compose logs -f
docker compose down
```

### 2.6 Checking disk space on the VM

To see how much disk space the VM has and what is using it:

**Overall disk usage (by filesystem)**
```bash
df -h
```
Shows each mount (e.g. `/`, `/mnt/lameness-data`) with **Size**, **Used**, **Avail**, and **Use%** in human-readable units.

**Largest directories under a path**
```bash
# Home and project (replace with your path)
du -sh ~/* 2>/dev/null
du -sh /path/to/vision-sam3-yolo-lameless/* 2>/dev/null

# Top-level usage under current directory
du -h --max-depth=1 . 2>/dev/null | sort -hr | head -20
```

**Docker-specific space**
```bash
docker system df
```
Shows space used by **Images**, **Containers**, and **Local Volumes** (and **Build Cache** if present).

### 2.7 Freeing disk space — remove everything from Docker

To free disk space on the VM (e.g. before shutting down the instance or to reclaim space after testing), remove all Docker containers, images, volumes, and build cache. **This deletes all app data stored in Docker volumes** (database, NATS, Qdrant, and any data under `./data` that is not on a host mount).

From the project root on the VM:

**1. Stop and remove the stack and its volumes**
```bash
cd vision-sam3-yolo-lameless
docker compose down -v
```
The `-v` flag removes named volumes (e.g. `postgres-data`, `nats-data`, `qdrant-data`).

**2. Remove all Docker resources on the system**
```bash
# Remove all stopped containers, unused images, and unused volumes
docker system prune -a -f --volumes
```
- `-a`: remove all unused images, not just dangling ones  
- `-f`: no confirmation prompt  
- `--volumes`: remove unused volumes  

**3. (Optional) Remove build cache**
```bash
docker builder prune -a -f
```

**4. Check space reclaimed**
```bash
docker system df
```

To start fresh after cleanup, run `./scripts/deploy.sh` again (and re-initialize the DB and data as on first deploy).

**If your Docker build fails with "No space left on device"** (e.g. during `mamba env create` in rater-reliability or another service), the VM or machine running Docker has run out of disk. Free space with the steps above (`docker builder prune -a -f` and `docker system prune -a -f --volumes`), then retry the build. If the VM disk is small, use a larger root volume or attach a volume and move Docker’s data dir (see below).

### 2.8 Increase disk space for Docker

If the root disk is too small for building or running all images, give Docker more space in one of these ways.

**Option A: Add a dedicated volume for Docker (recommended on OpenStack)**

1. **In the Arbutus dashboard**
   - **Volumes → Create Volume** (e.g. 1000 GB, name `docker-data`).
   - After creation, **▼ → Attach Volume** and select your instance. Note the device (e.g. `/dev/vdc`; the actual device may vary, e.g. `/dev/sdb`).

2. **On the VM** — find the device, format it, and mount it:
   ```bash
   # List block devices to confirm the new volume (e.g. vdc, sdb)
   lsblk
   # Format (use the device you see, e.g. /dev/vdb)
   sudo mkfs.ext4 /dev/vdb
   sudo mkdir -p /mnt/lameness-data
   sudo mount /dev/vdb /mnt/lameness-data
   ```
   To mount automatically on boot, add to `/etc/fstab`:
   ```bash
   echo '/dev/vdb /mnt/lameness-data ext4 defaults 0 2' | sudo tee -a /etc/fstab
   ```

3. **Point Docker and containerd at the new location**
   - Stop services first:
   ```bash
   sudo systemctl stop docker
   sudo systemctl stop containerd
   ```
   - Create config and set Docker’s data root to the new mount:
   ```bash
   sudo mkdir -p /etc/docker
   echo '{"data-root": "/mnt/lameness-data/docker-data"}' | sudo tee /etc/docker/daemon.json
   sudo mkdir -p /mnt/lameness-data/docker-data
   ```
   - Move containerd data dir to the same volume (prevents extraction snapshot errors on `/var/lib/containerd`):
   ```bash
   sudo mkdir -p /mnt/lameness-data/containerd-data
   if [ -d /var/lib/containerd ] && [ ! -L /var/lib/containerd ]; then
     sudo rsync -aHAX /var/lib/containerd/ /mnt/lameness-data/containerd-data/
     sudo mv /var/lib/containerd /var/lib/containerd.bak.$(date +%s)
   fi
   sudo ln -sfn /mnt/lameness-data/containerd-data /var/lib/containerd
   ```
   - Start services again:
   ```bash
   sudo systemctl start containerd
   sudo systemctl start docker
   ```
   - Confirm Docker/containerd are using the volume-backed paths:
   ```bash
   docker info | grep "Docker Root Dir"
   sudo readlink -f /var/lib/containerd
   ```
   `Docker Root Dir` should point to `/mnt/lameness-data/docker-data`, and `/var/lib/containerd` should resolve to `/mnt/lameness-data/containerd-data`.
   You can then run `./scripts/deploy.sh --skip-build`.


## Summary checklist

| Step | Action |
|------|--------|
| 1 | Log in at https://arbutus.cloud.computecanada.ca/ |
| 2 | Create/select project, import SSH key, create security group (SSH + 3000, 8000) |
| 3 | Launch Ubuntu 22.04 instance with 8+ vCPUs, 32+ GB RAM (or GPU flavor if needed) |
| 4 | Associate a floating IP to the instance |
| 5 | (Optional) Create and attach a Cinder volume; format and mount on the VM |
| 6 | SSH into the VM, install Docker and Docker Compose |
| 7 | Clone repo, `cp env.example .env`, set `DEPLOY_HOST=<FLOATING_IP>` |
| 8 | Set `DOCKER_HUB_USER` in `.env`, then run `./scripts/deploy.sh --skip-build` |
| 9 | Open http://\<FLOATING_IP\>:3000 and log in with default credentials |

---

## References

- [Alliance Cloud documentation](https://docs.alliancecan.ca/wiki/Cloud_Quick_Start) — Creating a Linux VM and general cloud usage.
- [Arbutus dashboard](https://arbutus.cloud.computecanada.ca/) — OpenStack web UI.
- Project docs in this repo: `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/PIPELINES_DETAILED.md`.
