sudo yum install python37 python37-pip cmake3 gcc-c++ git
ln -s /usr/bin/cmake3 /usr/bin/cmake
sudo pip3 install asv 
git clone https://github.com/PennyLaneAI/benchmark.git
git config --global credential.helper cache

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH=/home/ec2-user/miniconda3/bin:$PATH

cd ~/benchmark
./customenv_build.sh