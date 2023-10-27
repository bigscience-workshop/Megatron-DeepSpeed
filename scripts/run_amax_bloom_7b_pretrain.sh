#!/bin/bash
### GPUS
GPUS=8

#RUN_HOSTS=(g0001 g0002 g0003 g0004 g0005 g0006 g0007 g0008 g0009 g0010 g0011 g0012 g0013 g0014 g0015 g0016 g0017 g0018 g0019 g0020 g0021 g0022 g0023 g0024 g0025 g0026 g0027 g0028 g0029 g0030 g0031 g0032 g0033 g0034 g0035 g0036 g0037 g0038 g0039 g0040 g0041 g0042 g0043 g0044 g0045 g0046 g0047 g0048 g0049 g0050 g0051 g0052 g0053 g0054 g0055 g0056 g0057 g0058 g0059 g0060 g0061 g0062)

RUN_HOSTS=(g0029 g0050)

### <E8><84><9A><E6><9C><AC><E5><90><8D><E7><A7><B0>
RANK_SCRIPT="examples_deepspeed/pretrain_bloom_distributed_amax_7b.sh"

### Job Path
JOB_PATH=`pwd`

### Job ID
JOB_ID=`date +"%y%m%d%H%M%S"`
mkdir ${JOB_ID}

### hosfile
HOSTFILE="${JOB_ID}/hostfile"

### <E8><8E><B7><E5><8F><96><E8><8A><82><E7><82><B9><E4><B8><BB><E6><9C><BA><E5><90><8D>
for i in "${RUN_HOSTS[@]}";
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

### <E8><AE><BE><E7><BD><AE><E4><B8><BB><E8><8A><82><E7><82><B9>,<E5><B0><86><E7><AC><AC><E4><B8><80><E4><B8><AA><E8><8A><82><E7><82><B9><E4><B8><BB><E6><9C><BA><E5><90><8D><E5><81><9A><E4><B8><BA> master <E5><9C><B0><E5><9D><80>.
MASTER_ADDR=${host[1]}

### Nodes
NODES="${#host[@]}"

### <E6><B8><85><E7><90><86><E5><8F><AF><E8><83><BD><E5><AD><98><E5><9C><A8><E7><9A><84><E6><AE><8B><E7><95><99><E8><BF><9B><E7><A8><8B>.
#/usr/bin/pkill -9 python
for((i=1;i<=${NODES};i++));
do
   node_host=${host[$i]}
   pdsh -w ssh:"${node_host}" "/usr/bin/pkill -9 python"
done

# port release may be delayed
sleep 3

### nodes gpus rank master_addr hostfile job_id
#bash ${RANK_SCRIPT} ${NODES} ${GPUS} 0 ${MASTER_ADDR} ${HOSTFILE} ${JOB_ID} &
for((i=1;i<=${NODES};i++));
do
   node_host=${host[$i]}
   node_rank=${rank[$i]}
   echo "nodes:${NODES}, host:${node_host}, node_rank:${node_rank}, master_addr:${MASTER_ADDR}"
   pdsh -w ssh:"${node_host}" "cd ${JOB_PATH} ; /bin/bash ${RANK_SCRIPT} ${NODES} ${GPUS} $node_rank ${MASTER_ADDR} ${HOSTFILE} ${JOB_ID}" &
done
wait
  