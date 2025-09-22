# Detect your Ubuntu repo name (ubuntu2204 or ubuntu2404)
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID} | tr -d .)

# Add NVIDIA CUDA repo + key
wget https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install Toolkit only (keeps your existing driver)
sudo apt install -y cuda-toolkit   # or a specific major: sudo apt install -y cuda-toolkit-13

# (Optional) If you also want NVIDIA driver via CUDA meta-pkg:
# sudo apt install -y cuda          # installs driver + toolkit

# Set CUDA_HOME and PATH/LD_LIBRARY_PATH (persistent)
echo 'export CUDA_HOME=/usr/local/cuda'          >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH'          >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi | head -n 3
echo $CUDA_HOME
