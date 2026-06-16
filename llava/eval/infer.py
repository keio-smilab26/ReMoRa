import argparse
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
import functools
import itertools
import random
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print
from llava.train.motion_vector_loader import MotionVectorLoader
from llava.train.gop_video_loader import GOPVideoLoader


warnings.filterwarnings("ignore")

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time


def sanitize_task_name(name: str):
    name = name.strip().lower()
    # simple normalization for common dataset names
    mapping = {
        'videomme': 'videomme',
        'video-mme': 'videomme',
        'video_mme': 'videomme',
        'longvideobench': 'longvideobench',
        'longvideo-bench': 'longvideobench',
        'long_video_bench': 'longvideobench',
    }
    return mapping.get(name, name)


def load_gop_video_eval(video_path, image_processor, mv_loader, max_i_frames=128, fps=16):
    """GOP-aware loader for evaluation to mirror training pipeline.

    Returns: (gop_data_dict, frame_time_str, video_time)
    gop_data_dict keys: i_frames (tensor FxCxHxW), motion_vectors (TxHxWx2), frame_types, i_frame_indices,
                        gop_boundaries, video_time, fps, num_i_frames, total_frames
    """
    # Open video for metadata/time
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()

    # Initialize GOP loader
    gop_loader = GOPVideoLoader(
        motion_vector_loader=mv_loader,
        max_i_frames=max_i_frames,
    )

    gop_data = gop_loader.load_gop_video(
        video_path=video_path,
        video_reader=vr,
        fps=fps,
        uniform_sample=True,
    )

    if gop_data is None:
        return None, None, video_time

    # Build frame time string for I-frames (aligned with mv fps)
    frame_times = []
    for i_idx in gop_data.i_frame_indices:
        time_pos = i_idx / fps
        frame_times.append(f"{time_pos:.2f}s")
    frame_time_str = ",".join(frame_times)

    # Create tensor pack (preprocess I-frames to CLIP pixel_values)
    hybrid = gop_loader.create_hybrid_tensor(gop_data, image_processor)
    gop_data_dict = {
        'i_frames': hybrid['i_frames'],
        'motion_vectors': hybrid['motion_vectors'],
        'frame_types': hybrid['frame_types'],
        'i_frame_indices': hybrid['i_frame_indices'],
        'gop_boundaries': hybrid['gop_boundaries'],
        'video_time': gop_data.video_time,
        'fps': fps,
        'num_i_frames': len(gop_data.i_frames),
        'total_frames': gop_data.total_frames,
    }

    # Reset reader
    vr.seek(0)

    return gop_data_dict, frame_time_str, video_time

def get_options_letter(len_options):
    if len_options==2:
        return '(A or B)'
    elif len_options==3:
        return '(A, B or C)'
    elif len_options==4:
        return '(A, B, C or D)'
    elif len_options==5:
        return '(A, B, C, D, or E)'
    else:
        raise NotImplementedError

def strip_thinking_content(text):
    """Remove Qwen3 thinking content from output.

    Qwen3 outputs <think>...</think> before the actual answer.
    This function extracts only the answer portion.
    """
    import re
    # Pattern to match <think>...</think> content (including multi-line)
    think_pattern = r'<think>.*?</think>\s*'
    # Remove thinking content
    result = re.sub(think_pattern, '', text, flags=re.DOTALL)
    return result.strip()


def extract_mcq_answer(text):
    """Extract MCQ answer from model output.

    Handles cases where model outputs valid answer followed by garbage.
    Examples:
        "B. five\n### for the video..." -> "B"
        "D. agree with one another\n###..." -> "D"
        "A" -> "A"
    """
    import re

    # First, remove any "###" garbage and everything after it
    if '###' in text:
        text = text.split('###')[0].strip()

    # Try to extract answer letter at the start
    # Pattern: letter optionally followed by . or ) and optional text
    match = re.match(r'^([A-E])[\.\)\s]', text)
    if match:
        return match.group(1)

    # Try single letter answer
    match = re.match(r'^([A-E])$', text.strip())
    if match:
        return match.group(1)

    # Return cleaned text if no letter found
    return text.strip()


def get_prompt(dataset_name, sample, conv_template="qwen_1_5", video_time=None, num_frames=None, frame_time=None):
    # conv_template is passed as parameter, use it directly
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    if dataset_name in ['VSI']:
        prompt += "These are frames of a video.\n"
        prompt += sample["question"] + "\n"
        if 'candidates' in sample:
            for op in sample["candidates"]:
                prompt += f"{op}\n"
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += "Please answer the question using a single word or phrase."
    elif dataset_name in ['MovieChat']:
        if video_time is None:
            prompt += "These are frames of a video.\n"
        if 'time' in sample:
            timestamp = round(sample['time']/sample['fps'], 2)
            prompt += f"At time {timestamp}s, "
        prompt += sample["question"] + "\n"
        prompt += "Please answer the question using a single word, phrase, or sentence."
        #prompt += "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    elif dataset_name in ['ActivityNetQA']:
        # Open-ended QA task - capitalize question and add question mark
        raw_question = sample["question"]
        question_text = raw_question.capitalize() + "?"
        prompt += question_text + "\n"
        prompt += "Answer the question using a single word or phrase."
    elif dataset_name in ['MSRVTT', 'MSVD']:
        # Open-ended video QA for MSRVTT-QA and MSVD-QA
        prompt += sample["question"] + "\n"
        prompt += "Answer the question using a single word or phrase."
    else:
        options_letter = get_options_letter(len(sample['candidates']))
        prompt += f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {options_letter} of the correct option.\n"
        prompt += sample["question"] + "\n"
        for op in sample["candidates"]:
            prompt += f"{op}\n"
        prompt += f"The best answer is:"
        
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def run(rank, world_size, args):
    # Properly initialize CUDA for each process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Clear CUDA cache to prevent memory issues
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    rank0_print("Loadind dataset from", args.data_path)
    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    # CRITICAL: Set fixed seed for reproducible shuffling across all ranks
    # Without this, each GPU process shuffles differently, causing misalignment!
    random.seed(42)
    random.shuffle(dataset)

    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[rank:num_samples:world_size]
    rank0_print(f"Total samples: {num_samples}")
    print(f"Samples in rank {rank}: {len(dataset)}")

    # Load video ID mapping if provided (for MSVD dataset)
    video_id_mapping = {}
    if args.video_mapping_file and os.path.exists(args.video_mapping_file):
        rank0_print(f"Loading video ID mapping from: {args.video_mapping_file}")
        with open(args.video_mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        youtube_filename, vid_id = parts
                        # Store mapping from vid#### to youtube_filename
                        video_id_mapping[vid_id] = youtube_filename
        rank0_print(f"Loaded {len(video_id_mapping)} video ID mappings")

    # Use explicit device mapping for multiprocessing
    if world_size > 1:
        device_map = {"": f"cuda:{rank}"}
    else:
        device_map = "auto"
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        model_path = args.model_path,
                                                        model_base = args.model_base,
                                                        model_name = args.model_name,
                                                        lora_alpha = args.lora_alpha,
                                                        torch_dtype="bfloat16",
                                                        device_map=device_map,
                                                        attn_implementation=None,
                                                        overwrite_config = {"temporal_pooling":args.temporal_pooling},
                                                    )
    model.eval()
    
    # Ensure model is on correct device
    if world_size > 1:
        model = model.to(f"cuda:{rank}")


    # Initialize motion vector loader if enabled
    mv_loader = None
    if getattr(args, 'use_gop_loading', False):
        # Determine motion vector directory
        mv_dir = args.motion_vector_dir
        if not mv_dir:
            # Build from root + task name
            task_dir = sanitize_task_name(args.dataset_name)
            mv_dir = os.path.join(args.motion_vector_root, task_dir)
        rank0_print(f"[GOP] Initializing MotionVectorLoader from: {mv_dir}")
        try:
            # Index json defaults to mv_dir/video_index.json
            mv_loader = MotionVectorLoader(hdf5_dir=mv_dir)
            rank0_print(f"Motion vectors ready from {mv_dir}")
        except Exception as e:
            rank0_print(f"[GOP] Failed to init MotionVectorLoader at {mv_dir}: {e}")
            mv_loader = None

    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        sample_save_path = f"{args.results_dir}/outputs/{sample['id']}.json"
        if os.path.exists(sample_save_path):
            try:
                with open(sample_save_path, 'r') as f:
                    sample = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                rank0_print(f"Warning: Failed to load cached result from {sample_save_path}: {e}")
                rank0_print("Regenerating result...")
                # Continue to regenerate the result below

        if not os.path.exists(sample_save_path) or "prediction" not in sample:
            # Apply video ID mapping if available
            video_filename = sample["video"]
            if video_id_mapping:
                # Extract vid number from filename (e.g., vid1451.avi -> vid1451)
                vid_name = os.path.splitext(video_filename)[0]
                if vid_name in video_id_mapping:
                    youtube_name = video_id_mapping[vid_name]
                    # Keep the original extension
                    ext = os.path.splitext(video_filename)[1]
                    video_filename = youtube_name + ext
                    rank0_print(f"Mapped {sample['video']} -> {video_filename}")

            video_path = os.path.join(args.video_root, video_filename)
            use_gop = bool(args.use_gop_loading) and (mv_loader is not None)
            # Use the correct device for the current rank
            device = f"cuda:{rank}" if world_size > 1 else ("cuda" if torch.cuda.is_available() else "cpu")

            if use_gop:
                # Try GOP-aware evaluation pipeline
                rank0_print(f"[GOP] Attempting GOP load for sample {sample.get('id', 'NA')} -> {os.path.basename(video_path)}")
                gop_pack, i_frame_time_str, video_time = load_gop_video_eval(
                    video_path,
                    image_processor,
                    mv_loader,
                    max_i_frames=args.max_i_frames,
                    fps=args.gop_fps,
                )

                if gop_pack is not None:
                    # Prepare images and MV maps
                    try:
                        rank0_print(
                            f"[GOP] Loaded: I-frames={gop_pack['num_i_frames']}, "
                            f"Total MV frames={gop_pack['total_frames']}, "
                            f"i_frames_tensor={tuple(gop_pack['i_frames'].shape)}, "
                            f"mv_tensor={tuple(gop_pack['motion_vectors'].shape)}"
                        )
                    except Exception:
                        pass
                    i_frames = gop_pack['i_frames'].to(device).bfloat16()
                    images = [i_frames]

                    # Compute and stash per-GOP MV maps on the core model
                    try:
                        mv = gop_pack['motion_vectors']  # torch tensor (T,H,W,2)
                        if mv.device.type != device:
                            mv = mv.to(device)
                        mags = (mv[..., 0] ** 2 + mv[..., 1] ** 2).sqrt()
                        per_gop = []
                        for start, end in gop_pack['gop_boundaries']:
                            if end > start:
                                seg = mags[start:end]
                                per_gop.append(seg.mean(dim=0))
                            else:
                                seg = mags[start:start+1]
                                per_gop.append(seg.mean(dim=0))
                        mv_map_tensor = torch.stack(per_gop, dim=0) if len(per_gop) > 0 else None
                        core = model.get_model()
                        if hasattr(core, "pending_mv_maps"):
                            core.pending_mv_maps = [mv_map_tensor] if mv_map_tensor is not None else None
                        if mv_map_tensor is not None:
                            rank0_print(f"[GOP] Prepared mv_map_tensor for model: {tuple(mv_map_tensor.shape)}")
                    except Exception as e:
                        rank0_print(f"[GOP] Failed to prepare MV maps: {e}")

                    # Prompt
                    if args.use_time_ins:
                        # Mirror training-style time instruction for GOP
                        prompt_question = get_prompt(
                            args.dataset_name,
                            sample,
                            conv_template=args.conv_template,
                            video_time=video_time,
                            num_frames=gop_pack['num_i_frames'],
                            frame_time=i_frame_time_str,
                        )
                    else:
                        prompt_question = get_prompt(args.dataset_name, sample, conv_template=args.conv_template)

                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

                    try:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            cont = model.generate(
                                input_ids,
                                images=images,
                                modalities=["gop_video"],
                                do_sample=False,
                                temperature=0,
                                max_new_tokens=args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                repetition_penalty=1.1,
                            )
                        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                        # Strip Qwen3 thinking content if present
                        text_outputs = strip_thinking_content(text_outputs)
                        # Extract MCQ answer (removes garbage like "###...")
                        text_outputs = extract_mcq_answer(text_outputs)
                        sample["prediction"] = text_outputs
                    except RuntimeError as e:
                        print(f"Error (GOP) processing sample {sample['id']} on rank {rank}: {e}")
                        # Clear cache and fall back to regular pipeline
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        use_gop = False
                else:
                    rank0_print(f"[GOP] GOP features not available for {os.path.basename(video_path)}; falling back to standard frames")

            if not use_gop:
                if bool(args.use_gop_loading) and (mv_loader is None):
                    rank0_print("[GOP] MV loader unavailable; using standard video pipeline")
                # Fallback to standard frame-based evaluation
                video,frame_time,video_time = load_video(video_path, args.max_frames_num, fps=1, force_sample=True)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
                images = [video]
                if args.use_time_ins:
                    prompt_question = get_prompt(args.dataset_name, sample, conv_template=args.conv_template, video_time=video_time, num_frames=args.max_frames_num, frame_time=frame_time)
                else:
                    prompt_question = get_prompt(args.dataset_name, sample, conv_template=args.conv_template)

                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

                try:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        cont = model.generate(
                            input_ids,
                            images=images,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            repetition_penalty=1.1,
                        )
                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                    # Strip Qwen3 thinking content if present
                    text_outputs = strip_thinking_content(text_outputs)
                    # Extract MCQ answer (removes garbage like "###...")
                    text_outputs = extract_mcq_answer(text_outputs)
                    sample["prediction"] = text_outputs
                except RuntimeError as e:
                    print(f"Error processing sample {sample['id']} on rank {rank}: {e}")
                    # Try to recover by clearing cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    sample["prediction"] = "ERROR"

            with open(sample_save_path, "w") as f:
                json.dump(sample, f, indent=4)
        
        result_list.append(sample)
        if "answer" in sample:
            print(cnt, "GT:", sample["answer"], "Pred:", sample["prediction"])
        else:
            print(cnt, "Pred:", sample["prediction"])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Run Inference")

    # Model
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--max_frames_num", type=int, default=128)
    parser.add_argument("--temporal_pooling", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--use_time_ins", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=None)
    # GOP/MV evaluation
    parser.add_argument("--use_gop_loading", action="store_true", help="Enable GOP-aware evaluation with motion vectors")
    parser.add_argument("--gop_fps", type=int, default=16, help="FPS for motion vectors (4/8/16)")
    parser.add_argument("--max_i_frames", type=int, default=64, help="Max I-frames to sample for GOP")
    parser.add_argument("--motion_vector_root", type=str, default="DATAS/motion_vectors", help="Root dir for motion vectors")
    parser.add_argument("--motion_vector_dir", type=str, default=None, help="Explicit dir for motion vectors (overrides root/task)")

    # Data
    parser.add_argument("--dataset_name", type=str, default="VideoMME")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the formatted evaluation JSON")
    parser.add_argument("--video_root", type=str, required=True, help="Directory containing the evaluation videos")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to write predictions to")
    parser.add_argument("--video_mapping_file", type=str, default=None, help="Path to video ID mapping file (youtube_id vid_number format)")
    parser.add_argument("--test_ratio", type=float, default=1)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--cals_acc", action="store_true")

    args = parser.parse_args()
    if args.model_base == "None":
        args.model_base = None

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)


    if args.multiprocess:
        # Set spawn method to avoid CUDA context issues
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set
        
        print(f"started benchmarking")
        n_gpus = torch.cuda.device_count()
        world_size = min(n_gpus, 8)  # Limit to 8 GPUs max for stability
        print("World size", world_size)
        
        # Use multiprocessing spawn for CUDA compatibility
        with mp.get_context("spawn").Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        print("finished running")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args)
    

    if args.cals_acc:
        results = {"all": {"correct": 0, "total": 0}}
        for sample in result_list:
            if "answer" not in sample:
                continue
            results["all"]["total"] += 1
            if "question_type" in sample:
                if sample["question_type"] not in results:
                    results[sample["question_type"]] = {"correct": 0, "total": 0}
                results[sample["question_type"]]["total"] += 1
                
            if sample["answer"].lower()==fuzzy_matching(sample["prediction"]).lower():
                results["all"]["correct"] += 1
                if "question_type" in sample:
                    results[sample["question_type"]]["correct"] += 1

        for key in results:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]

        print(results)

        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
