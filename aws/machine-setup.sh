sudo yum install python37 python37-pip cmake3 gcc-c++ git python3-devel
ln -s /usr/bin/cmake3 /usr/bin/cmake
sudo python3 -m pip install asv 
git clone https://github.com/PennyLaneAI/benchmark.git
git config --global credential.helper cache


cd ~/benchmark
./customenv_build.sh