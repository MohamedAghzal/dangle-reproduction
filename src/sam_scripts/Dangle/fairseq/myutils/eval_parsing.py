import os, shutil
src_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
src_path = f"{src_path}/fairseq/fairseq/data/language_pair_dataset_NO_INPUT_FEEDING.py"
src_path = f"/home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset_NO_INPUT_FEEDING.py"
# src_path = f"/home/sam/Dangle/fairseq/fairseq/data/language_pair_dataset_INPUT_FEEDING.py"
dst_path = f"{os.path.dirname(src_path)}/language_pair_dataset.py"
os.system(f'rm {dst_path}; cp {src_path} {dst_path}')
# shutil.copyfile(src_path, dst_path)

from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.data import encoders
from fairseq.logging.meters import StopwatchMeter, TimeMeter

import torch
import sys, os, math, glob, re, random
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from collections import Counter

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='semantic_parsing')
args = options.parse_args_and_arch(parser)
print(f"  args: {args}")

omsgs = []
for k, v in args.__dict__.items():
    omsgs.append(f"  eval_parsing.py: args[{str(k):>80}]: {v}")
print('*'*60, '\n'.join(sorted(omsgs, key = lambda x: x.replace(' ','').lower(),)), '*'*60, sep='\n',)
# exit()

# Setup task
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)

import fairseq
from fairseq import tasks
from fairseq.tasks import semantic_parsing
from fairseq.tasks.semantic_parsing import SemanticParsingTask
task: fairseq.tasks.semantic_parsing.SemanticParsingTask = task
print(f"  task: {task}")
for td_idx,(key,value) in enumerate(task.__dict__.items()):
    print(f"  td[{key}]: {value}")
test: fairseq.data.language_pair_dataset.LanguagePairDataset = task.__dict__['datasets']['test']


print(f"*" * 60,)
# print(f"  task['datasets']: {task['datasets']}")
# print(f"*" * 60,)
# test = task['datasets']['test']
# for test_idx, (_test) in enumerate(test):
#     print(f"  test[{test_idx}]: {_test}")
    
# exit()
# print(f"  args.gen_subset: {args.gen_subset}")
# test = task.gen_subset
# for test_idx, (_test) in enumerate(test):
#     print(f"  test[{test_idx}]: {_test}")
    
# exit()


omsgs = []
for k, v in task.__dict__.items():
    omsgs.append(f"  eval_parsing.py: task[{str(k):>80}]: {v}")
print('*'*60, '\n'.join(sorted(omsgs, key = lambda x: x.replace(' ','').lower(),)), '*'*60, sep='\n',)

print(args)
print(args.max_sentences)
print(args.max_tokens)
# Set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary
src_dict = task.source_dictionary

last_was_unk = False
for tgt_dict_idx, (_tgt_dict) in enumerate(tgt_dict):
    # print(f"  tgt_dict[{tgt_dict_idx}]: {_tgt_dict}")
    if "<unk>" in _tgt_dict:
        if last_was_unk:
            break
        last_was_unk = True
    
last_was_unk = False
for src_dict_idx, (_src_dict) in enumerate(src_dict):
    # print(f"  src_dict[{src_dict_idx}]: {_src_dict}")
    if "<unk>" in _src_dict:
        if last_was_unk:
            break
        last_was_unk = True


if args.results_path is not None:
    os.makedirs(args.results_path, exist_ok=True)
    output_path = os.path.join(args.results_path, 'generate-{}.txt.tmp'.format(args.gen_subset))
    if os.path.exists(output_path):
        print(f"  eval_parsing.py: output_path[{output_path}] already exists, exiting!")
        exit()
    output_file = open(output_path, 'w', buffering=1, encoding='utf-8')
else:
    output_file = sys.stdout

code_file_path = os.path.join(args.data, "{}.word-predicate.code".format(args.gen_subset))
if os.path.exists(code_file_path):
    with open(code_file_path) as f:
        codes = [line.strip() for line in f.readlines()]
else:
    codes = None


use_cuda = torch.cuda.is_available() and not args.cpu
# Load ensemble
print('loading model(s) from {}'.format(args.path))
try:
    args.path = f"{args.path.replace('best', 'last')}"
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
except:
    args.path = f"{args.path.replace('last', 'best')}"
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
# Optimize ensemble for generation
for model in models:
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

# Load dataset (possibly sharded)
print(f"  task: {task}")
itr = task.get_batch_iterator(
    dataset=task.dataset(args.gen_subset),
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    ),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    num_shards=args.num_shards,
    shard_id=args.shard_id,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)
progress = progress_bar.progress_bar(
    itr,
    log_format=args.log_format,
    log_interval=args.log_interval,
    default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
)

for idx, sample in enumerate(itr):
    print(f"*" * 60,)
    for sample_idx,(key,value) in enumerate(sample.items()):
        print(f"  sample[{key}]: {value}")
    if idx > 5:
        break
print(f"*" * 60,)
# exit()


# Initialize generator
gen_timer = StopwatchMeter()
generator = task.build_generator(models, args)

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
import numpy as np
def decode_src(x):
    return decode_fn(x[0])
    try:
        try:
            return tokenizer.decode(x, skip_special_tokens=True).strip()
        except Exception as e0:
            try:
                return tokenizer.decode(x[0], skip_special_tokens=True).strip()
            except Exception as e1:
                print(f"  Error in decode_src():  x: {x}, e0: {e0}, e1: {e1}")
                raise e1
    except Exception as e:
        print(f"  Error in decode_src():  x: {x},  e: {e}")
        raise e

def p(*args, **kwargs):
    return
    kwargs.pop('file', None)
    print(*args, **kwargs)

code_results = {}
partial_accuracies = []
partial_accuracies_dict = {}
full_accuracies_dict = {}

type_num, type_correct_num = Counter(), Counter()
correct, wrong, isum = 0, 0, 0
correct_examples, wrong_examples = [], []
gid = 0
from tqdm import tqdm
for eid, sample in enumerate(tqdm(progress)):
# for eid, sample in enumerate(progress):
    # if np.sum(sample['net_input']['src_lengths'].item()) <= 3:
    #     continue
    # src_tokens = sample['net_input']['src_tokens'].cpu().numpy()
    # if 3 in (src_tokens):
    #     continue
    # target = sample['target'].cpu().numpy()
    # if 3 in (target):
    #     continue
    # prev_output_tokens = np.int32([2] * (len(target) + 1))
    # prev_output_tokens = np.array(prev_output_tokens)
    # sample['net_input']['prev_output_tokens'] = torch.Tensor([prev_output_tokens]).long().cuda()
    # print(f"  src_tokens: {src_tokens}")
    # assert(np.sum(sample['net_input']['src_lengths'].item()) > 2)

    sample = utils.move_to_cuda(sample) if use_cuda else sample
    if 'net_input' not in sample:
        continue
    # for sample_idx,(key,value) in enumerate(sample.items()):
    #     print(f"  sample[{key}]: {value}")
    src = sample['net_input']['src_tokens'].cpu().numpy()
    decoded_src = decode_src(src)
    # print(f"  decoded_src: {decoded_src}")
    # exit()
        
    sample_id = sample['id'].cpu().numpy()
    # print(f"  sample_id: {sample_id}")
    # exit()

    prefix_tokens = None
    if args.prefix_size > 0:
        prefix_tokens = sample['target'][:, :args.prefix_size]
    
    hypos = task.inference_step(generator, models, sample, prefix_tokens)

    gen_timer.start()
    hypos = task.inference_step(generator, models, sample, prefix_tokens)
    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
    gen_timer.stop(num_generated_tokens)


    print_pred_still = 1
    for i, sample_id in enumerate(sample['id'].tolist()):
        has_target = sample['target'] is not None

        # Remove padding
        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
        target_tokens = None
        if has_target:
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

        # Either retrieve the original sentences or regenerate them from tokens.
        if src_dict is not None:
            src_str = src_dict.string(src_tokens)
        else:
            src_str = ""

        if has_target:
            target_str = tgt_dict.string(
                target_tokens,
                escape_unk=True,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
        if args.source_bpe_decode:
            detok_src_str = decode_fn(src_str)
        else:
            detok_src_str = src_str

        if args.target_bpe_decode and has_target:
            detok_target_str = decode_fn(target_str)
        else:
            detok_target_str = target_str
    
    
        if not args.quiet:
            print('\n', file=output_file)
            print(str(gid), file=output_file)
           
            if codes is not None:
                print('code : {}'.format(codes[sample_id]), file=output_file)

            if src_dict is not None:
                print('S-token\t{}'.format(src_tokens), file=output_file)
                print('S\t{}'.format(src_str), file=output_file)
                print('S-decode\t{}\t{}'.format(sample_id, detok_src_str), file=output_file)
            if has_target:
                print('T-token\t{}'.format(target_tokens), file=output_file)
                print('T\t{}'.format(target_str), file=output_file)
                print('T-decode\t{}\t{}'.format(sample_id, detok_target_str), file=output_file)


        # Process top predictions
        for j, hypo in enumerate(hypos[i][:args.nbest]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'],
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
            if args.target_bpe_decode:
                detok_hypo_str = decode_fn(hypo_str)
            else:
                detok_hypo_str = hypo_str

            hypo_str = tgt_dict.string(
                    hypo_tokens,
                    escape_unk=True,
                    extra_symbols_to_ignore={
                        generator.eos,
                    }
                )

            t_hypo = hypo_tokens
            t_target = target_tokens
            if len(t_hypo) > len(t_target):
                t_hypo = t_hypo[:len(t_target)]
            len_diff = max(0, len(t_target) - len(t_hypo))
            if len(t_hypo) < len(t_target):
                t_hypo = torch.cat([t_hypo, torch.zeros(len(t_target) - len(t_hypo)).long()])



            s0 = (torch.sum(t_hypo == target_tokens, dtype=torch.float32).cpu().numpy() - len_diff)/ (len(t_target))
            s0 = max(0, (torch.sum(t_hypo == target_tokens, dtype=torch.float32).cpu().numpy() - len_diff)) / (len(t_target))
            accuracy = max(0,s0)
            partial_accuracies_dict['average'] = partial_accuracies_dict.get('average', []) + [accuracy]
            partial_accuracies_dict[codes[sample_id]] = partial_accuracies_dict.get(codes[sample_id], []) + [accuracy]
            if print_pred_still:
                print_pred_still = 0
                def p2(*args, **kwargs):
                    print('-', *args, **kwargs,)
                p2(f"-" * 60,)
                p2(f"            src_str: {src_str}")  
                # p2(f"             sample_id: {sample_id}")
                # p2(f"  attempt/prediction #: {j}")
                p2(f"                  code: {codes[sample_id]}")
                p2(f"      detok_target_str: {detok_target_str}")
                p2(f"        detok_hypo_str: {detok_hypo_str}")
                p2(f"              hypo_str: {hypo_str}")
                p2(f"                 match: {int(detok_hypo_str == detok_target_str)}")
                p2(f"-" * 60,)
            code = codes[sample_id]
            correct = int(detok_hypo_str == detok_target_str)
            code_results[code] = code_results.get(code, []) + [correct,]

            
            

            if codes is not None:
                if detok_target_str == detok_hypo_str:
                    type_correct_num[codes[sample_id]] += 1
                full_accuracies_dict["average"] = full_accuracies_dict.get("average", []) + [int(detok_hypo_str == detok_target_str)]
                full_accuracies_dict[codes[sample_id]] = full_accuracies_dict.get(codes[sample_id], []) + [int(detok_hypo_str == detok_target_str)]

                type_num[codes[sample_id]] += 1
                
            if not args.quiet:
                if detok_target_str == detok_hypo_str:
                    p('Correct', file=output_file)
                else:
                    p('Incorrect', file=output_file)

                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                p('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                # detokenized hypothesis
                p('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                p('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ), file=output_file)

                if args.print_alignment:
                    p('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                    ), file=output_file)

                if args.print_step:
                    p('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                if getattr(args, 'retain_iter_history', False):
                    for step, h in enumerate(hypo['history']):
                        _, h_str, _ = utils.post_process_prediction(
                            hypo_tokens=h['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=None,
                            align_dict=None,
                            tgt_dict=tgt_dict,
                            remove_bpe=None,
                        )
                        p('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

            # Score only the top hypothesis
            if has_target and j == 0:
                if detok_target_str == detok_hypo_str:
                    correct += 1
                    # correct_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                    # print("correct_examples")
                else:
                    wrong += 1
                    # wrong_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                isum += 1

        gid += 1
    print(f"*" * 60,)
    print(f"  eid: {eid}")
    print(f"  len(progress): {len(progress)}")
    print(f"  acc: {[(type_correct_num[t]/type_num[t]) for t in type_num]}",)
    print(f"*" * 20,)
    for partial_accuracies_dict_idx,(key,value) in enumerate(full_accuracies_dict.items()):
        print(f"     full {key:>80}: {np.mean(value):0.3f}")       
    print(f"*" * 20,)
    for partial_accuracies_dict_idx,(key,value) in enumerate(partial_accuracies_dict.items()):
        print(f"  partial {key:>80}: {np.mean(value):0.3f}")       
    # if eid > 100:
    #     break
            

print(f"*" * 60,)
print(f"  eid: {eid}")
print(f"  acc: {[(type_correct_num[t]/type_num[t]) for t in type_num]}",)
print(f"*" * 20,)
for partial_accuracies_dict_idx,(key,value) in enumerate(full_accuracies_dict.items()):
    print(f"     full {key:>80}: {np.mean(value):0.3f}")       
print(f"*" * 20,)
for partial_accuracies_dict_idx,(key,value) in enumerate(partial_accuracies_dict.items()):
    print(f"  partial {key:>80}: {np.mean(value):0.3f}")  

os.makedirs(args.results_path, exist_ok=True)
output_path2 = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
if not os.path.exists(output_path2):
    output_file2 = open(output_path, 'w', buffering=1, encoding='utf-8')
    print("Model : "+args.path+" ;  Accuracy : "+str(correct/isum), file=output_file2)
    print("Code type Accuracy : "+"  ;  ".join([t+" : "+str(type_correct_num[t]/type_num[t]) for t in type_num]), file=output_file2)
    print(f"  acc: {[(type_correct_num[t]/type_num[t]) for t in type_num]}", file=output_file2)

    for partial_accuracies_dict_idx,(key,value) in enumerate(full_accuracies_dict.items()):
        print(f"     full {key:>80}: {np.mean(value):0.3f}", file=output_file2)
    for partial_accuracies_dict_idx,(key,value) in enumerate(partial_accuracies_dict.items()):
        print(f"  partial {key:>80}: {np.mean(value):0.3f}", file=output_file2)
    
          
# print(f"  acc: {[(type_correct_num[t]/type_num[t]) for t in type_num]}",)
# for partial_accuracies_dict_idx,(key,value) in enumerate(partial_accuracies_dict.items()):
#     print(f"  {key:>80}: {np.mean(value):0.3f}")       


# for partial_accuracies_dict_idx,(key,value) in enumerate(partial_accuracies_dict.items()):
#     print(f"  partial_accuracies_dict[{key}]: {np.mean(value)}", file=output_file2)


exit()
from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.data import encoders
from fairseq.logging.meters import StopwatchMeter, TimeMeter

import torch
import sys, os, math, glob, re, random
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from collections import Counter

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='semantic_parsing')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)

print(args)
print(args.max_sentences)
print(args.max_tokens)
# Set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary
src_dict = task.source_dictionary


if args.results_path is not None:
    os.makedirs(args.results_path, exist_ok=True)
    output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
    output_file = open(output_path, 'w', buffering=1, encoding='utf-8')
else:
    output_file = sys.stdout
print(f"  writing to {output_file}")

code_file_path = os.path.join(args.data, "{}.word-predicate.code".format(args.gen_subset))
if os.path.exists(code_file_path):
    with open(code_file_path) as f:
        codes = [line.strip() for line in f.readlines()]
else:
    codes = None

print(f"  eval(args.model_overrides): {eval(args.model_overrides)}")


use_cuda = torch.cuda.is_available() and not args.cpu
# Load ensemble
print('loading model(s) from {}'.format(args.path))
args.path = f"{str(args.path).replace('checkpoint_last', 'checkpoint_best')}"
models, _model_args = checkpoint_utils.load_model_ensemble(
    utils.split_paths(args.path),
    arg_overrides=eval(args.model_overrides),
    task=task,
    suffix=getattr(args, "checkpoint_suffix", ""),
)
# Optimize ensemble for generation
for model in models:
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

# Load dataset (possibly sharded)
itr = task.get_batch_iterator(
    dataset=task.dataset(args.gen_subset),
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    ),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    num_shards=args.num_shards,
    shard_id=args.shard_id,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)
progress = progress_bar.progress_bar(
    itr,
    log_format=args.log_format,
    log_interval=args.log_interval,
    default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
)


# Initialize generator
gen_timer = StopwatchMeter()
generator = task.build_generator(models, args)

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
# use sentencepiece model
import transformers
gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
# bpe_model_path = '/home/sam/Dangle/fairseq/COGS/data/full_data/multiModel.model' 
# vocab_path = '/home/sam/Dangle/fairseq/COGS/data/full_data/multiModel.vocab'
# tokenizer = transformers.BertTokenizer(vocab_file=vocab_path)

tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    try:
        return tokenizer.decode(x)[:, 1:-1]
    except:
        try:
            return tokenizer.decode([x])
        except:
            return tokenizer.decode(x)
    raise Exception("tokenizer.decode failed")

    return x

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


def p(*args, **kwargs):
    return
    kwargs.pop('file', None)
    print(*args, **kwargs)

type_num, type_correct_num = Counter(), Counter()
correct, wrong, isum = 0, 0, 0
correct_examples, wrong_examples = [], []
gid = 0
accuracies = []
for eid, sample in enumerate(progress):
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    if 'net_input' not in sample:
        continue

    prefix_tokens = None
    if args.prefix_size > 0:
        prefix_tokens = sample['target'][:, :args.prefix_size]
    #     print(f"  prefix_tokens.shape: {prefix_tokens.shape}")
    # else:
    #     print(f"  prefix_tokens: {prefix_tokens}")

    # for sample_idx,(key,value) in enumerate(sample.items()):
    #     try:
    #         print(f"  sample[{key}]: {value.shape}")
    #     except:
    #         print(f"  sample[{key}]: {value}")
        
    hypos = task.inference_step(generator, models, sample, prefix_tokens)
    target = sample['target']

    

    # scores = []
    # tokes = []
    # for hypos_idx, (_hypos) in enumerate(hypos):
    #     tokens = _hypos[0]['tokens']
    #     tokes.append(tokens)
    #     scores.append(_hypos[0]['score'])

    # for tgt, hypo in zip(target, tokes):
    #     hypo_len = len(hypo)
    #     target_len = len(tgt)
    #     miss_ctr = abs(hypo_len - target_len)
    #     if target_len < hypo_len:
    #         tgt = torch.cat((tgt, torch.zeros(hypo_len - target_len).long().cuda()))
    #     elif target_len > hypo_len:
    #         hypo = torch.cat((hypo, torch.zeros(target_len - hypo_len).long().cuda()))
    #     else:
    #         miss_ctr = 0
    #     accuracy = max( 0, ((hypo == tgt).sum().item() - miss_ctr)) / (len(tgt))
    #     accuracies.append(accuracy)
    # continue
    # print(f"  accuracy: {accuracy}")
    # exit()


    gen_timer.start()
    hypos = task.inference_step(generator, models, sample, prefix_tokens)
    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
    gen_timer.stop(num_generated_tokens)

    if eid % 10 == 0:
        print("Finished %d examples " % eid)

    for i, sample_id in enumerate(sample['id'].tolist()):
        has_target = sample['target'] is not None
        print(f"*" * 60,)
        print(f"  i: {i}")
        print(f"*" * 60,)

        # Remove padding
        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
        target_tokens = None
        if has_target:
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

        # Either retrieve the original sentences or regenerate them from tokens.
        if src_dict is not None:
            src_str = src_dict.string(src_tokens)
        else:
            src_str = ""

        if has_target:
            target_str = tgt_dict.string(
                target_tokens,
                escape_unk=True,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
        if args.source_bpe_decode:
            detok_src_str = decode_fn(src_str)
        else:
            detok_src_str = src_str

        if args.target_bpe_decode and has_target:
            detok_target_str = decode_fn(target_str)
        else:
            detok_target_str = target_str
    
    
        # if not args.quiet:
        if True:
            p('\n', file=output_file)
            p(str(gid), file=output_file)
           
            if codes is not None:
                p('code : {}'.format(codes[sample_id]), file=output_file)

            if src_dict is not None:
                p('S-token\t{}'.format(src_tokens), file=output_file)
                p('S\t{}'.format(src_str), file=output_file)
                p('S-decode\t{}\t{}'.format(sample_id, detok_src_str), file=output_file)
            if has_target:
                p('T-token\t{}'.format(target_tokens), file=output_file)
                p('T\t{}'.format(target_str), file=output_file)
                p('T-decode\t{}\t{}'.format(sample_id, detok_target_str), file=output_file)



        # Process top predictions
        for j, hypo in enumerate(hypos[i][:args.nbest]):
            p(f"*" * 60,)
            p(f"  j: {j}")
            print(f"*" * 60,)
            p, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'],
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
            hypo_str = tgt_dict.string(
                    hypo_tokens,
                    escape_unk=True,
                    extra_symbols_to_ignore={
                        generator.eos,
                    }
                )
            detok_hypo_str = decode_fn(hypo_str)
            # if args.target_bpe_decode:
            #     detok_hypo_str = decode_fn(hypo_str)
            # else:
            #     detok_hypo_str = hypo_str

            if codes is not None:
                if detok_target_str == detok_hypo_str:
                    type_correct_num[codes[sample_id]] += 1
                type_num[codes[sample_id]] += 1
                
            # if not args.quiet:
            if True:
                if detok_target_str == detok_hypo_str:
                    p('Correct', file=output_file)
                else:
                    p('Incorrect', file=output_file)

                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                p('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                # detokenized hypothesis
                p('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                p('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ), file=output_file)

                if args.print_alignment:
                    p('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                    ), file=output_file)

                if args.print_step:
                    p('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                if getattr(args, 'retain_iter_history', False):
                    for step, h in enumerate(hypo['history']):
                        _, h_str, _ = utils.post_process_prediction(
                            hypo_tokens=h['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=None,
                            align_dict=None,
                            tgt_dict=tgt_dict,
                            remove_bpe=None,
                        )
                        p('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

            # Score only the top hypothesis
            if has_target and j == 0:
                if detok_target_str == detok_hypo_str:
                    correct += 1
                    correct_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                    p("correct_examples")
                else:
                    wrong += 1
                    # wrong_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                isum += 1

        gid += 1
        exit()
import numpy as np
accuracy = np.mean(accuracies)
print("Model : "+args.path+" ;  Accuracy : "+str(accuracy), )
print("Model : "+args.path+" ;  Accuracy : "+str(accuracy), file=output_file)
print("Accuracy : "+str(accuracy), file=output_file)

# print("Model : "+args.path+" ;  Accuracy : "+str(correct/isum), )
# if codes is not None:
#     print("Code type Accuracy : "+"  ;  ".join([t+" : "+str(type_correct_num[t]/type_num[t]) for t in type_num]), )

# print("Model : "+args.path+" ;  Accuracy : "+str(correct/isum), file=output_file)
# if codes is not None:
#     print("Code type Accuracy : "+"  ;  ".join([t+" : "+str(type_correct_num[t]/type_num[t]) for t in type_num]), file=output_file)

                    