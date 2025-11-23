sudo apt update
sudo apt upgrade -y

# Install required headers & dev packages
sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# check the version of the kernel you have 
uname -r

# head to the NVIDIA CUDA download page and get the latest driver link for your OS version
https://docs.nvidia.com/cuda/cuda-installation-guide-linux

# pick the repository and then install the version stable as of now is 

sudo apt --purge remove '*nvidia*' '*cuda*'
sudo apt autoremove -y
sudo apt clean
sudo reboot

# we can list the available GPUs here 
sudo ubuntu-drivers list --gpgpu

# recommended to install this version of the driver 
sudo apt install -y nvidia-driver-535-server nvidia-utils-535-server
sudo reboot 

# install only the toolkit 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# check the installation of the nvidia-smi driver through this 
nvidia-smi