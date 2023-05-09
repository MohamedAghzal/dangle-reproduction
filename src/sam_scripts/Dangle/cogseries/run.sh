source ~/.bashrc
cd /home/sam/Dangle
#conda activate dangle
prefix="/home/sam/Dangle/cogseries"


suffix="transformer_absolute 0 2 /home/sam/Dangle/fairseq/COGS"


if [ "${2}" == "dangle" ] ; then
	suffix="transformer_dangle_absolute 0 2 /home/sam/Dangle/fairseq/COGS"
fi


series_name="monolingualCOGS"
if [ "${3}" == "ml" ] ; then
	series_name="MULTILINGUALCOGS"
fi
series_name="${series_name}_${1}"


suffix="${suffix} ${series_name}"

first_char=${1:0:1}

cmd="${prefix}/${1}.sh ${suffix} "
if [ "${first_char}" == "m" ] ; then
	cmd="${prefix}/${1}.sh ${suffix}"
fi
if [ "${first_char}" == "c" ] ; then
	cmd="${prefix}/${1}.sh ${suffix}"
fi
if [ "${first_char}" == "e" ] ; then
	cmd="${prefix}/${1}.sh ${suffix}"
fi
if [ "${first_char}" == "n" ] ; then
	cmd="${prefix}/${1}.sh ${suffix}"
fi
#${prefix}/c0.sh ${suffix}
#${prefix}/c1.sh ${suffix}
#${prefix}/c2.sh ${suffix}

#${prefix}/${cmd}.sh ${suffix}
echo $cmd
$cmd
exit 0

${prefix}/d0.sh transformer_dangle_absolute 0 2 /home/sam/Dangle/fairseq/COGS &
${prefix}/b0.sh ${suffix} &
${prefix}/b2.sh ${suffix} &
${prefix}/c0.sh ${suffix} &
${prefix}/c2.sh ${suffix} &
${prefix}/d1.sh transformer_dangle_absolute 0 2 /home/sam/Dangle/fairseq/COGS &
${prefix}/e0.sh ${suffix} &
${prefix}/e1.sh ${suffix} &
${prefix}/e2.sh ${suffix} &
${prefix}/m0.sh ${suffix} &
${prefix}/m2.sh ${suffix} &
${prefix}/n2.sh ${suffix} &
${prefix}/n3.sh ${suffix} &

#${prefix}/m1.sh ${suffix}
#${prefix}/m2.sh ${suffix}

exit 0

# choices:
# e1, e2 # 2L Dec, 4L Dec, 128H

conda activate dangle
/home/sam/Dangle/cogseries/run.sh b0 dangle ml
/home/sam/Dangle/cogseries/run.sh b1 dangle ml
/home/sam/Dangle/cogseries/run.sh b2 dangle ml
/home/sam/Dangle/cogseries/run.sh b0 nd ml
/home/sam/Dangle/cogseries/run.sh b1 nd ml
/home/sam/Dangle/cogseries/run.sh b2 nd ml

