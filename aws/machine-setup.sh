sudo yum install python37 python37-pip
sudo python3 -m pip install asv virtualenv
sudo yum install git
git clone https://github.com/PennyLaneAI/benchmark.git

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

cd ~/benchmark
./customenv_build.sh