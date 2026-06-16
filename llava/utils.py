import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
except ImportError:
    av = None
    print("Warning: pyav not installed. Some video processing functions may not work.")

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError as e:
    VideoReader = None
    cpu = None
    DECORD_AVAILABLE = False
    import sys
    if "decord" not in sys.modules:
        print(f"Warning: decord not installed. Please install it with: pip install decord. Error: {e}")

def process_video_with_decord(video_file, data_args):
    # Try importing again in case it wasn't available during module initialization
    global VideoReader, cpu, DECORD_AVAILABLE
    if not DECORD_AVAILABLE:
        try:
            from decord import VideoReader, cpu
            DECORD_AVAILABLE = True
        except ImportError:
            pass
    
    if VideoReader is None or cpu is None:
        raise ImportError("decord is not installed. Please install it with: pip install decord")
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    # Use actual video FPS for timestamps; sampling step may differ
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def process_video_with_gop(
    video_file: str, 
    motion_vector_loader,
    data_args,
    max_i_frames: int = 64,
    fps: int = 16
) -> Tuple[Dict[str, Any], float, str, int]:
    """
    Process video with GOP awareness, loading I-frames as images and P/B frames as motion vectors.
    
    Args:
        video_file: Path to video file
        motion_vector_loader: Instance of MotionVectorLoader
        data_args: Data arguments containing processing parameters
        max_i_frames: Maximum number of I-frames to sample
        fps: FPS rate for motion vector extraction
    
    Returns:
        Tuple of (gop_data_dict, video_time, frame_time_str, num_frames)
    """
    from llava.train.gop_video_loader import GOPVideoLoader, GOPVideoData
    from PIL import Image
    
    # Initialize video reader
    global VideoReader, cpu, DECORD_AVAILABLE
    if not DECORD_AVAILABLE:
        try:
            from decord import VideoReader, cpu
            DECORD_AVAILABLE = True
        except ImportError:
            pass
    
    if VideoReader is None or cpu is None:
        raise ImportError("decord is not installed. Please install it with: pip install decord")
    
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    # Initialize GOP loader
    gop_loader = GOPVideoLoader(
        motion_vector_loader=motion_vector_loader,
        max_i_frames=max_i_frames
    )
    
    # Load GOP-aware video data
    gop_data = gop_loader.load_gop_video(
        video_path=video_file,
        video_reader=vr,
        fps=fps,
        uniform_sample=True
    )
    
    if gop_data is None:
        # Fallback to regular video processing if GOP loading fails
        print(f"Warning: GOP loading failed for {video_file}, falling back to regular processing")
        # Return None to indicate GOP loading failed, let caller handle fallback
        return None, video_time, "", 0
    
    # Create frame time string for I-frames
    frame_times = []
    for i_idx in gop_data.i_frame_indices:
        time_pos = i_idx / fps
        frame_times.append(f"{time_pos:.2f}s")
    frame_time_str = ",".join(frame_times)
    
    # Create hybrid tensor representation
    hybrid_tensors = gop_loader.create_hybrid_tensor(gop_data, data_args.image_processor)

    # Check if tensor creation failed
    if hybrid_tensors is None:
        print(f"Warning: Failed to create hybrid tensors for {video_file}, falling back to regular processing")
        return None, video_time, "", 0

    # Package data for model
    gop_data_dict = {
        'i_frames': hybrid_tensors['i_frames'],  # Tensor of I-frame images
        'motion_vectors': hybrid_tensors['motion_vectors'],  # Motion vectors for all frames
        'frame_types': hybrid_tensors['frame_types'],  # Frame type indices
        'i_frame_indices': hybrid_tensors['i_frame_indices'],  # Positions of I-frames
        'gop_boundaries': hybrid_tensors['gop_boundaries'],  # GOP boundary information
        'video_time': video_time,
        'fps': fps,
        'num_i_frames': len(gop_data.i_frames),
        'total_frames': gop_data.total_frames,
        'modality': 'gop_video'  # New modality type
    }
    
    num_frames = len(gop_data.i_frames)  # Report number of I-frames
    
    # Clean up
    vr.seek(0)
    
    return gop_data_dict, video_time, frame_time_str, num_frames
