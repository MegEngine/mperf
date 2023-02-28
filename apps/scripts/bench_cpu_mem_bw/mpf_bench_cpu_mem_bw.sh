#!/bin/bash -e

export PATH=$PATH:`pwd`

echo "usage: ./mpf_bench_cpu_mem_bw.sh len(4) parallel_num(1) device_id(id0[,id1,...]) extend(1)"

MB=$1
PARALLEL_NUM=$2
CORE=$3
EXTEND=$4

WARMUP=5
REPEAT=1000
CORE_LIST=$(echo $CORE | sed 's/,/_/g')

# Figure out how big we can go for stuff that wants to use all of memory.
ALL="512 1k 2k 4k 8k 16k 32k 64k 128k 256k 512k 1m 2m 4m"
i=8
while [ $i -le $MB ]
do
	ALL="$ALL ${i}m"
	i=`expr $i \* 2`
done

echo \[`date`] 1>&2
echo \[MAX MB: ${MB}MB] 1>&2
echo \[PARALLEL_NUM: ${PARALLEL_NUM}] 1>&2

echo "----test bandwidth------"
whereis mperf_cpu_mem_bw
rm -f "./bw_mem_core_${CORE_LIST}_report.txt"
echo "" 1>&2
echo "ALL: $ALL"

# continous memory access patterns
echo "libc-bcopy" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i bcopy; done; echo "" 1>&2

echo "libc-bzero" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i bzero; done; echo "" 1>&2

echo "copy" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i fcp; done; echo "" 1>&2

echo "read" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i frd; done; echo "" 1>&2

echo "write" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i fwr; done; echo "" 1>&2

echo "read-write-same-addr" 1>&2
for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i frdwr; done; echo "" 1>&2


if [ $EXTEND -eq 1 ]; then
	# strided memory access patterns
	echo "copy-stride-4" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i scp; done; echo "" 1>&2

	echo "read-stride-4" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i srd; done; echo "" 1>&2

	echo "write-stride-4" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i swr; done; echo "" 1>&2

	# random memory access patterns
	echo "random-mem-rd" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i rnd_rd; done; echo "" 1>&2

	echo "random-mem-wr" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i rnd_wr; done; echo "" 1>&2

	# micro-kernel cases
	echo "triad" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i triad; done; echo "" 1>&2

	echo "add1" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i add1; done; echo "" 1>&2

	echo "add2" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i add2; done; echo "" 1>&2

	echo "mla" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i mla; done; echo "" 1>&2

	echo "sum" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i sum; done; echo "" 1>&2

	echo "dot" 1>&2
	for i in $ALL; do mperf_cpu_mem_bw -P $PARALLEL_NUM -W $WARMUP -N $REPEAT -C $CORE $i dot; done; echo "" 1>&2

	# Add more useful micro-kernel cases here...


	# Add more cases here...(eg. gather/scatter) 


fi
