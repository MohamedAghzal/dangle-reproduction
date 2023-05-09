#!/bin/bash
# clear; python /home/sam/Dangle/textDump.py; cd /home/sam/Dangle/fairseq/; .././pp_cogs.sh /home/sam/Dangle/fairseq/COGS; cd ..;

#source ~/.bashrc
#conda activate dangle

MODEL=$1 # transformer_relative | transformer_absolute | transformer_dangle_relative | transformer_dangle_absolute | roberta_dangle
SEED=$2
RECURSION=$3 # 2 | 3 | 4 | 5
DATADIR=$4 # path/to/cogs
prefix=$5

max_split_size_mb=128
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:${max_split_size_mb}

###
# DEPR
lr="5e-5"
maxEpoch="--max-epoch 100"
saveInterval="--save-interval 15"

maxEpoch=""
saveInterval=""
saveIntervalUpdate="--save-interval-updates 1000"
maxUpdate="--max-update 10000"

maxEpoch="--max-epoch 30"
saveInterval="--save-interval 5"
saveIntervalUpdate=""
maxUpdate=""

###

############################################################
run_training=1
run_eval=1
export_eval=1
############################################
dO="0.1"
ffn="1024"
encL="0"
decL="4"
embed="256"
encAttHeads="2"
decAttHeads="2"
weightDecay="0.1"
modelNameTag="__${encL}-${decL}L_${dO}dropout_${embed}H_${ffn}FFN_${encAttHeads}-${decAttHeads}AH_${weightDecay}WD"

encoder_embed_dim="--encoder-embed-dim ${embed}"
encoder_embed_dim=""
dropout="--dropout ${dO}"
attentionDropout="--attention-dropout ${dO}"
modelKWARGS="--dropout ${dO} \
--attention-dropout ${dO} \
--encoder-embed-dim ${embed} \
--decoder-embed-dim ${embed} \
--encoder-ffn-embed-dim ${ffn} \
--decoder-ffn-embed-dim ${ffn} \
--encoder-layers ${encL} \
--decoder-layers ${decL} \
--encoder-attention-heads ${encAttHeads} \
--decoder-attention-heads ${decAttHeads} \
--weight-decay ${weightDecay} \
"
maxSentences=64
lrScheduler="inverse_sqrt"
lrScheduler="cosine"
max_len=128
kwargs="--min-lr 1e-5 --max-lr 1e-3 --lr-period-updates 2000 --lr-shrink 0.9 --max-source-positions ${max_len} --max-target-positions ${max_len} --required-batch-size-multiple 8"
############################################################
#usage: fairseq-train [-h] [--no-progress-bar] [--log-interval N] [--log-format {json,none,simple,tqdm}] [--tensorboard-logdir DIR] [--seed N] [--cpu] [--tpu] [--bf16] [--fp16] [--memory-efficient-bf16] [--memory-efficient-fp16] [--fp16-no-flatten-grads] [--fp16-init-scale FP16_INIT_SCALE] [--fp16-scale-window FP16_SCALE_WINDOW] [--fp16-scale-tolerance FP16_SCALE_TOLERANCE] [--min-loss-scale D]
#                     [--threshold-loss-scale THRESHOLD_LOSS_SCALE] [--user-dir USER_DIR] [--empty-cache-freq EMPTY_CACHE_FREQ] [--all-gather-list-size ALL_GATHER_LIST_SIZE] [--model-parallel-size N] [--checkpoint-suffix CHECKPOINT_SUFFIX] [--quantization-config-path QUANTIZATION_CONFIG_PATH] [--profile]
#                     [--criterion {label_smoothed_cross_entropy,binary_cross_entropy,cross_entropy,masked_lm,nat_loss,adaptive_loss,sentence_ranking,legacy_masked_lm_loss,label_smoothed_cross_entropy_with_alignment,sentence_prediction,composite_loss,vocab_parallel_cross_entropy}] [--tokenizer {space,moses,nltk}] [--bpe {hf_byte_bpe,fastbpe,bytes,bert,subword_nmt,gpt2,byte_bpe,sentencepiece,characters}]
#                     [--optimizer {lamb,adamax,sgd,adadelta,adam,adafactor,adagrad,nag}] [--lr-scheduler {polynomial_decay,tri_stage,triangular,cosine,fixed,inverse_sqrt,reduce_lr_on_plateau}] [--task TASK] [--num-workers N] [--skip-invalid-size-inputs-valid-test] [--max-tokens N] [--max-sentences N] [--required-batch-size-multiple N] [--dataset-impl FORMAT] [--data-buffer-size N] [--train-subset SPLIT]
#                     [--valid-subset SPLIT] [--validate-interval N] [--fixed-validation-seed N] [--disable-validation] [--max-tokens-valid N] [--max-sentences-valid N] [--curriculum N] [--distributed-world-size N] [--distributed-rank DISTRIBUTED_RANK] [--distributed-backend DISTRIBUTED_BACKEND] [--distributed-init-method DISTRIBUTED_INIT_METHOD] [--distributed-port DISTRIBUTED_PORT] [--device-id DEVICE_ID]
#                     [--distributed-no-spawn] [--ddp-backend {c10d,no_c10d}] [--bucket-cap-mb MB] [--fix-batches-to-gpus] [--find-unused-parameters] [--fast-stat-sync] [--broadcast-buffers] [--distributed-wrapper {DDP,SlowMo}] [--slowmo-momentum SLOWMO_MOMENTUM] [--slowmo-algorithm {LocalSGD,SGP}] [--localsgd-frequency LOCALSGD_FREQUENCY] [--nprocs-per-node N] [--arch ARCH] [--max-epoch N] [--max-update N]
#                     [--stop-time-hours N] [--clip-norm NORM] [--sentence-avg] [--update-freq N1,N2,...,N_K] [--lr LR_1,LR_2,...,LR_N] [--min-lr LR] [--use-bmuf] [--save-dir DIR] [--restore-file RESTORE_FILE] [--reset-dataloader] [--reset-lr-scheduler] [--reset-meters] [--reset-optimizer] [--optimizer-overrides DICT] [--save-interval N] [--save-interval-updates N] [--keep-interval-updates N] [--keep-last-epochs N]
#                     [--keep-best-checkpoints N] [--no-save] [--no-epoch-checkpoints] [--no-last-checkpoints] [--no-save-optimizer-state] [--best-checkpoint-metric BEST_CHECKPOINT_METRIC] [--maximize-best-checkpoint-metric] [--patience N] [--activation-fn {relu,gelu,gelu_fast,gelu_accurate,tanh,linear}] [--dropout D] [--attention-dropout D] [--activation-dropout D] [--encoder-embed-path STR] [--encoder-embed-dim N]
#                     [--encoder-ffn-embed-dim N] [--encoder-layers N] [--encoder-attention-heads N] [--encoder-normalize-before] [--encoder-learned-pos] [--decoder-embed-path STR] [--decoder-embed-dim N] [--decoder-ffn-embed-dim N] [--decoder-layers N] [--decoder-attention-heads N] [--decoder-learned-pos] [--decoder-normalize-before] [--decoder-output-dim N] [--share-decoder-input-output-embed] [--share-all-embeddings]
#                     [--no-token-positional-embeddings] [--adaptive-softmax-cutoff EXPR] [--adaptive-softmax-dropout D] [--layernorm-embedding] [--no-scale-embedding] [--no-cross-attention] [--cross-self-attention] [--encoder-layerdrop D] [--decoder-layerdrop D] [--encoder-layers-to-keep ENCODER_LAYERS_TO_KEEP] [--decoder-layers-to-keep DECODER_LAYERS_TO_KEEP] [--quant-noise-pq D] [--quant-noise-pq-block-size D]
#                     [--quant-noise-scalar D] [--use-rel-pos] [--max-relative-position MAX_RELATIVE_POSITION] [--target-horizon TARGET_HORIZON] [--glove-scale GLOVE_SCALE] [--adam-betas B] [--adam-eps D] [--weight-decay WD] [--use-old-adam] [--warmup-updates N] [--warmup-init-lr LR] [--max-lr LR] [--t-mult LR] [--lr-period-updates LR] [--lr-shrink LS] [-s SRC] [-t TARGET] [--load-alignments] [--left-pad-source BOOL]
#                     [--left-pad-target BOOL] [--max-source-positions N] [--max-target-positions N] [--upsample-primary UPSAMPLE_PRIMARY] [--truncate-source] [--num-batch-buckets N] [--eval-accuracy] [--eval-accuracy-print-samples]



DATA="${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}"
BPE_DATA="${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}-bpe"


baseDir="/home/sam/Dangle"
DangleDir="/home/sam/Dangle"
saveDir="${baseDir}/model_saves"
tbDir_train="${saveDir}/tensorboard/TRAINING"
tbDir_eval="${saveDir}/tensorboard/EVALUATION"
mkdir -p ${tbDir_train} ${tbDir_eval} ${saveDir} ${DangleDir} ${baseDir}

fp16="--fp16"
fp16="--memory-efficient-fp16"
maxTokens="1024"
maxTokens="65536"
maxTokens="--max-tokens 4096"
maxTokens=""
warmupUpdates="4000"
warmupUpdates="50"
warmupUpdates="50"
optimizer="adam"
clipNorm="1.0"
workers=""


shared_prefix="\
fairseq-train $DATA"

uni_shared=" \
--task semantic_parsing \
--dataset-impl raw \
--share-decoder-input-output-embed \
--optimizer ${optimizer} \
--clip-norm ${clipNorm} \
--fp16-init-scale 32 \
--keep-best-checkpoints 1 \
--no-scale-embedding \
--no-epoch-checkpoints \
--lr ${lr} \
${kwargs} \
${fp16} \
--lr-scheduler ${lrScheduler} \
--warmup-updates ${warmupUpdates} \
--criterion cross_entropy \
${maxTokens} \
${maxUpdate} \
--seed ${SEED} \
${workers} \
--validate-interval 1 \
${saveIntervalUpdate} \
${saveInterval} \
${maxEpoch} \
--max-sentences ${maxSentences} \
--skip-invalid-size-inputs-valid-test \
${modelKWARGS} \
"

shared="${uni_shared} --encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt"
shared_eval="python ${DangleDir}/myutils/eval_parsing.py ${DATA} --gen-subset test --dataset-impl raw --quiet ${maxTokens} --max-sentences 1 --seed ${SEED} ${workers} --bpe gpt2  --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe --source-bpe-decode --target-bpe-decode"

shared="${uni_shared}"
shared_eval="python ${DangleDir}/myutils/eval_parsing.py ${DATA} --gen-subset test --dataset-impl raw --quiet ${maxTokens} --max-sentences 1 --seed ${SEED} ${workers} "
shared_eval="python ${DangleDir}/myutils/eval_parsing.py ${DATA} --gen-subset test --dataset-impl raw --quiet ${maxTokens} --max-sentences 100 --seed ${SEED} ${workers} --skip-invalid-size-inputs-valid-test "




if [[ "$MODEL" == transformer* && ! -f "glove.840B.300d.txt" ]] ; then
	wget https://nlp.stanford.edu/data/glove.840B.300d.zip 
	unzip glove.840B.300d.zip
fi

if [ "$MODEL" == "transformer_relative" ] ; then
	arch="transformer_glove_rel_pos"

elif [ "$MODEL" == "transformer_absolute" ] ; then
	arch="transformer_glove"


elif [ "$MODEL" == "transformer_dangle_relative" ] ; then
	arch="transformer_dangle_relative_enc_glove"

elif [ "$MODEL" == "transformer_dangle_absolute" ] ; then
	arch="transformer_dangle_enc_glove"


elif [ "$MODEL" == "roberta_dangle" ] ; then
	arch="roberta_dangle"
	
	# Set more variables
	name="${prefix}COGS_${MODEL}_recursion-${3}_seed-${SEED}"
	WORKDIR=${saveDir}/${name}

	# Define and run train_cmd
	train_cmd="fairseq-train $BPE_DATA --tensorboard-logdir ${tbDir_train}/${name} --arch ${arch} --save-dir ${WORKDIR} --roberta-path "roberta.base" ${shared}"
	CUDA_VISIBLE_DEVICES=0 ${train_cmd}

	# Define and run eval_cmd
	eval_cmd="${shared_eval} --path "${WORKDIR}/checkpoint_last.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR}"
	CUDA_VISIBLE_DEVICES=0 ${eval_cmd}

	output_shared_eval="python ${DangleDir}/pushEvalToTB.py ${DATA} --gen-subset test --dataset-impl raw --quiet --max-sentences ${maxSentences} --path "${WORKDIR}/checkpoint_last.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR} --max-tokens 1024 --seed 0 ${workers} "
	CUDA_VISIBLE_DEVICES=0 ${output_shared_eval}


	WORKDIR=~/cogs_roberta_dangle_seed${SEED}
	if [ ! -d "roberta.base" ] ; then
		wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
		tar -xzvf roberta.base.tar.gz
		wget -N encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
		wget -N vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
	fi
	
	##train
	CUDA_VISIBLE_DEVICES=0  \
	 \
	--task semantic_parsing --dataset-impl raw --arch roberta_dangle  --tensorboard-logdir /home/sam/Dangle/tb/roberta_dangle \
	--optimizer adam --clip-norm 1.0 \
	--lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 1024 --max-update 35000 \
	--save-dir $WORKDIR \
	 \
	--no-epoch-checkpoints \
	--no-scale-embedding \
	--bpe gpt2 --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe \
	--seed $SEED ;
	#evalute
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $BPE_DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 1 \
	--bpe gpt2  --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe --source-bpe-decode --target-bpe-decode ;
	exit 0

elif [ "$MODEL" == "mbert_dangle" ] ; then
	arch="roberta_dangle"
	
	# Set more variables
	name="cogs_${MODEL}_recursion-${3}_seed-${SEED}"
	WORKDIR=${saveDir}/${name}

	# Define and run train_cmd
	train_cmd="fairseq-train $BPE_DATA --tensorboard-logdir ${tbDir_train}/${name} --arch ${arch} --save-dir ${WORKDIR} --roberta-path "roberta.base" ${shared}"
	CUDA_VISIBLE_DEVICES=0 ${train_cmd}

	# Define and run eval_cmd
	eval_cmd="${shared_eval} --path "${WORKDIR}/checkpoint_best.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR}"
	CUDA_VISIBLE_DEVICES=0 ${eval_cmd}

	output_shared_eval="python ${DangleDir}/pushEvalToTB.py ${DATA} --gen-subset test --dataset-impl raw --quiet --max-sentences ${maxSentences} --path "${WORKDIR}/checkpoint_last.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR} --max-tokens 1024 --seed 0 ${workers} "
	CUDA_VISIBLE_DEVICES=0 ${output_shared_eval}


	WORKDIR=~/cogs_roberta_dangle_seed${SEED}
	if [ ! -d "roberta.base" ] ; then
		wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
		tar -xzvf roberta.base.tar.gz
		wget -N encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
		wget -N vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
	fi
	
	##train
	CUDA_VISIBLE_DEVICES=0  \
	 \
	--task semantic_parsing --dataset-impl raw --arch roberta_dangle  --tensorboard-logdir /home/sam/Dangle/tb/roberta_dangle \
	--optimizer adam --clip-norm 1.0 \
	--lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 1024 --max-update 35000 \
	--save-dir $WORKDIR \
	 \
	--no-epoch-checkpoints \
	--no-scale-embedding \
	--bpe gpt2 --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe \
	--seed $SEED ;
	#evalute
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $BPE_DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 1 \
	--bpe gpt2  --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe --source-bpe-decode --target-bpe-decode ;
	exit 0


else
	echo "ERROR: Unrecognized Model Type"
	exit 1
fi

if [ "$MODEL" == "roberta_dangle" ] ; then
	exit 0
fi
# Set more variables

name="${prefix}_${MODEL}_recursion-${3}_seed-${SEED}_${modelNameTag}"
WORKDIR=${saveDir}/${name}

#################################################
# Define and run train_cmd
train_cmd="${shared_prefix} --tensorboard-logdir ${tbDir_train}/${name} --arch ${arch} --save-dir ${WORKDIR} ${shared}"
if [ "$run_training" == "1" ] ; then
	rm /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset.py
	cp /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset_INPUT_FEEDING.py /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset.py 
	CUDA_VISIBLE_DEVICES=0 ${train_cmd}
fi
#################################################

# Define and run eval_cmd
eval_cmd="${shared_eval} --path "${WORKDIR}/checkpoint_best.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR}"
if [ "$run_eval" == "1" ] ; then 
	rm /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset.py
	cp /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset_NO_INPUT_FEEDING.py /home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset.py 
	CUDA_VISIBLE_DEVICES=0 ${eval_cmd}
fi
#################################################

output_shared_eval="python ${DangleDir}/pushEvalToTB.py ${DATA} --gen-subset test --dataset-impl raw --quiet --max-sentences ${maxSentences} --path "${WORKDIR}/checkpoint_best.pt" --tensorboard-logdir ${tbDir_eval}/${name} --results-path ${WORKDIR} --max-tokens 1024 --seed 0 ${workers} "
if [ "$export_eval" == "1" ] ; then 
	CUDA_VISIBLE_DEVICES=0 ${output_shared_eval}
fi

