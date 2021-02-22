# AWS procedure

Github action `configure-aws-credentials`

AWS CLI: start instance
```
start 
```

SSH commands to instance Github actions `ssh-action`



### ssh into instance

```
ssh -i /path/my-key-pair.pem my-instance-user-name@my-instance-public-dns-name
```

### in the instance

```
sudo yum install python37 python37-pip
sudo python3 -m pip install asv
sudo yum install git
git clone https://github.com/PennyLaneAI/benchmark.git
asv run --bench CircuitEvaluation

```
