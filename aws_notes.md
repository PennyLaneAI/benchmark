# AWS Procedure

This document outlines the specifics of running the benchmarks on AWS.  

Currently, it is a set of research notes.


## Instance Types

General purpose vs. compute optimized vs. memory optimized, storage optimized.

While the most *important* thing we will be doing is computing, we will have to install packages many times, requiring good network.  Therefore, and also because its easiest, let's just go for a *General Purpose instance*.

### Core types

We have a choice of hardware:
* Intel
* AMD
* Graviton2

Graviton2 seems to be cheapest, but not indicative of what user's probably have access to.  

### Spot instance

Spot instances occur with spare computing capacity.  Much cheaper

"also vailable to run for a defined duration-- in hourly increments up to six hours in length".  We could request an hour of defined duration?

Off peak hours have additional savings: run benchmarks every Sunday.

#### Spot Instance Requests

Need to specify:
* duration: 1hour?

can terminate early?

Create a Launch template

Custom AMI?

One-time spot requests, the memory will be deleted after.  We could make a persistent one?  Or have the hard drive memory persistent?

### ssh

We run commands on the instance using ssh.  

Github action "Run SSH command"


## Security and key pairs

AWS stores a public key, and you need the private key on your computer in order to access instance

Can specify protocols, ports,and source IP ranges that can reach instance

## Github and CLI

Github action `configure-aws-credentials`

Github already has the AWS CLI

## limiting variability

Do I need to run this command each time?

We need to fix at P1 to disable turbo boost and lower processor speed variability

Open `/etc/default/grub` with sudo and edit these lines to edit the c-state

```
GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,115200n8 net.ifnames=0 biosdevname=0 nvme_core.io_timeout=4294967295 intel_idle.max_cstate=1"
GRUB_TIMEOUT=0
```

Rebuild the boot configuration

```
grub2-mkconfig -o /boot/grub2/grub.cfg
```

Reboot Instance
```
sudo reboot
```

Disable turbo boost
```
sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
```

We won't be doing so, but can re-enable turbo boost by
```
 sudo sh -c "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo"
```

## Typical Run

Install asv

clone repository

disable turbo boost

```
asv run NEW --steps 3 --bench some-keyword --cpu-affinity 0 --machine instance_name --skip-exisiting --interleave-processes
```


# Random notes

Do I send machine information to github repository and use that each time?  Probably best.  Then just have to specify `--machine MACHINE` on each `asv run` to link the machine that we routinely use.

How many steps?

Pin to multiple CPU's?

## Terminology

*P-state*: optimization of voltage and CPU frequency

*C-state*: optimization of the power consumption if a core does not have to execute any instructions

*Amazon Linux 2*: next-generation Amazon Linux operating system, current LTS

*Amazon Linux AMI*: deprecating linux instance

*hypervisor*: virtual machine monitor/ VMM, software that creates and runs virtual machines

*AWS Nitro System*: dedicated hardware + lightweight hypervisor

*Graviton2*: amazon special processors

*Security Group*: virtual firewall to control incoming and outgoing traffic

*Processor affinity*: CPU pining, enables the binding and unbinding of a process or a thread to a CPU or a range of CPUs

*EBS*: Elastic Block Storage persistent block storage`

## Links

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/processor_state_control.html#baseline-perf