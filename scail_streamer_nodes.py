import os
import json
import math
import subprocess
import shutil
import numpy as np
import cv2
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import folder_paths
except ImportError:
    class FolderPathsMock:
        def get_output_directory(self):
            return os.path.abspath("./output")
        def get_temp_directory(self):
            return os.path.abspath("./temp")
    folder_paths = FolderPathsMock()


def get_ffmpeg_path():
    """Locate the ffmpeg executable on Windows, Linux, or within ComfyUI portable environment."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    base_cwd = os.getcwd()
    candidates = [
        os.path.join(base_cwd, "ffmpeg.exe"),
        os.path.join(base_cwd, "python_embeded", "Scripts", "ffmpeg.exe"),
        os.path.join(base_cwd, "bin", "ffmpeg.exe")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "ffmpeg"


def resolve_project_path(rel_or_abs_path):
    clean_path = (rel_or_abs_path or "scail_project").strip()
    if os.path.isabs(clean_path):
        base_dir = os.path.normpath(clean_path)
    else:
        base_dir = os.path.normpath(os.path.join(folder_paths.get_output_directory(), clean_path))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def tensor_to_numpy(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor, dtype=np.float32)


def numpy_to_tensor(array):
    if torch is not None:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return array


def reinhard_color_transfer_np(target_rgb, ref_rgb):
    target_u8 = (target_rgb * 255.0).clip(0, 255).astype(np.uint8)
    ref_u8 = (ref_rgb * 255.0).clip(0, 255).astype(np.uint8)

    target_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    res_lab = np.zeros_like(target_lab)
    for c in range(3):
        t_mean, t_std = np.mean(target_lab[:, :, c]), np.std(target_lab[:, :, c])
        r_mean, r_std = np.mean(ref_lab[:, :, c]), np.std(ref_lab[:, :, c])

        if t_std < 1e-6:
            res_lab[:, :, c] = target_lab[:, :, c]
        else:
            res_lab[:, :, c] = ((target_lab[:, :, c] - t_mean) * (r_std / t_std)) + r_mean

    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    res_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return res_rgb


# =====================================================================
# Node 1: SCAIL_VideoChunkIterator
# =====================================================================
class SCAIL_VideoChunkIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The complete source video frames."}),
                "output_folder": ("STRING", {
                    "default": "scail_project",
                    "multiline": False,
                    "tooltip": "Relative path inside ComfyUI's output folder."
                }),
                "frame_length": ("INT", {"default": 81, "min": 16, "max": 1024, "step": 1}),
                "overlap_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),
                "mode": (["Auto Queue Loop", "Manual Segment Index", "Reset Session State"], {"default": "Auto Queue Loop"}),
                "manual_segment_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "audio": ("AUDIO", {})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "BOOLEAN", "INT", "INT", "INT", "STRING", "FLOAT", "AUDIO")
    RETURN_NAMES = (
        "pose_chunk", "previous_frames", "segment_index", "total_segments",
        "is_last_segment", "video_frame_offset", "width_32", "height_32",
        "output_folder", "fps", "audio"
    )
    FUNCTION = "iterate_chunk"
    CATEGORY = "SCAIL_Streamer"

    def iterate_chunk(self, images, output_folder, frame_length, overlap_count, mode, manual_segment_index, fps=24.0, audio=None):
        total_frames = images.shape[0]
        orig_height = images.shape[1]
        orig_width = images.shape[2]

        width_32 = (orig_width // 32) * 32
        height_32 = (orig_height // 32) * 32

        stride = frame_length - overlap_count
        if stride <= 0:
            raise ValueError(f"frame_length ({frame_length}) must be strictly greater than overlap_count ({overlap_count}).")

        if total_frames <= frame_length:
            total_segments = 1
        else:
            total_segments = 1 + math.ceil((total_frames - frame_length) / stride)

        base_dir = resolve_project_path(output_folder)
        cache_dir = os.path.join(base_dir, "cache")
        frames_dir = os.path.join(base_dir, "temp_frames")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        state_file = os.path.join(cache_dir, "iterator_state.json")

        if mode == "Reset Session State":
            current_index = 0
            with open(state_file, "w") as f:
                json.dump({"current_index": 0, "total_segments": total_segments}, f)
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)
            for cache_f in ["previous_frames.npy", "last_exit_frame.npy", "manifest.json"]:
                cp = os.path.join(cache_dir, cache_f)
                if os.path.exists(cp):
                    os.remove(cp)
        elif mode == "Manual Segment Index":
            current_index = manual_segment_index
        else:
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        state_data = json.load(f)
                    current_index = state_data.get("current_index", 0)
                except Exception:
                    current_index = 0
            else:
                current_index = 0

        if current_index >= total_segments:
            current_index = total_segments - 1

        is_last_segment = (current_index >= total_segments - 1)
        video_frame_offset = current_index * stride

        start_frame = current_index * stride
        end_frame = start_frame + frame_length

        if end_frame <= total_frames:
            pose_chunk = images[start_frame:end_frame]
        else:
            available = images[start_frame:total_frames]
            missing = frame_length - available.shape[0]
            if missing > 0:
                if torch is not None and isinstance(available, torch.Tensor):
                    pad = available[-1:].repeat(missing, 1, 1, 1)
                    pose_chunk = torch.cat([available, pad], dim=0)
                else:
                    pad = np.repeat(available[-1:], missing, axis=0)
                    pose_chunk = np.concatenate([available, pad], axis=0)
            else:
                pose_chunk = available

        cache_prev_file = os.path.join(cache_dir, "previous_frames.npy")
        if current_index == 0 or not os.path.exists(cache_prev_file):
            if torch is not None:
                previous_frames = torch.zeros((1, images.shape[1], images.shape[2], images.shape[3]), dtype=images.dtype)
            else:
                previous_frames = np.zeros((1, images.shape[1], images.shape[2], images.shape[3]), dtype=np.float32)
        else:
            cached_arr = np.load(cache_prev_file)
            previous_frames = numpy_to_tensor(cached_arr)

        next_index = current_index + 1 if not is_last_segment else 0
        with open(state_file, "w") as f:
            json.dump({
                "current_index": next_index,
                "total_segments": total_segments,
                "last_processed": current_index
            }, f)

        print(f"[SCAIL_VideoChunkIterator] Processing Segment {current_index + 1}/{total_segments}. Is Last: {is_last_segment}")

        return (
            pose_chunk, previous_frames, current_index, total_segments,
            is_last_segment, video_frame_offset, width_32, height_32,
            output_folder, fps, audio
        )


# =====================================================================
# Node 2: SCAIL_DiskVideoCollector
# =====================================================================
class SCAIL_DiskVideoCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_images": ("IMAGE", {"tooltip": "Decoded image frames from VAEDecode."}),
                "output_folder": ("STRING", {"default": "scail_project", "multiline": False}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "is_last_segment": ("BOOLEAN", {"default": False}),
                "overlap_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),
                "apply_color_transfer": ("BOOLEAN", {"default": True}),
                "save_video_every_segment": ("BOOLEAN", {"default": True, "tooltip": "Compile MP4 after each segment so you can preview progress immediately."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "output_filename": ("STRING", {"default": "final_scail_video.mp4"}),
                "cleanup_temp_frames": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO", {})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("cached_last_frames", "final_video_path", "status")
    FUNCTION = "collect_and_save"
    CATEGORY = "SCAIL_Streamer"
    OUTPUT_NODE = True

    def collect_and_save(
        self, generated_images, output_folder, segment_index, is_last_segment,
        overlap_count, apply_color_transfer, save_video_every_segment, fps,
        output_filename, cleanup_temp_frames=False, audio=None
    ):
        base_dir = resolve_project_path(output_folder)
        frames_dir = os.path.join(base_dir, "temp_frames")
        cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        ref_frame_path = os.path.join(cache_dir, "last_exit_frame.npy")
        prev_frames_path = os.path.join(cache_dir, "previous_frames.npy")
        manifest_path = os.path.join(cache_dir, "manifest.json")

        img_np = tensor_to_numpy(generated_images)

        if segment_index == 0 or not os.path.exists(manifest_path):
            current_frame_counter = 0
            for existing_f in os.listdir(frames_dir):
                if existing_f.endswith(".png"):
                    try:
                        os.remove(os.path.join(frames_dir, existing_f))
                    except Exception:
                        pass
        else:
            try:
                with open(manifest_path, "r") as f:
                    m = json.load(f)
                current_frame_counter = m.get("total_saved_frames", 0)
            except Exception:
                current_frame_counter = 0

        if segment_index == 0:
            keep_frames = img_np
        else:
            keep_frames = img_np[overlap_count:]
            if apply_color_transfer and os.path.exists(ref_frame_path):
                try:
                    ref_frame = np.load(ref_frame_path)
                    corrected_list = []
                    for i in range(keep_frames.shape[0]):
                        corr = reinhard_color_transfer_np(keep_frames[i], ref_frame)
                        corrected_list.append(corr)
                    keep_frames = np.stack(corrected_list, axis=0)
                except Exception as e:
                    print(f"[SCAIL_DiskVideoCollector] Color transfer warning: {e}")

        last_exit_frame = img_np[-1]
        np.save(ref_frame_path, last_exit_frame)

        overlap_frames = img_np[-overlap_count:]
        np.save(prev_frames_path, overlap_frames)

        for i in range(keep_frames.shape[0]):
            frame_u8 = (keep_frames[i] * 255.0).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(frame_u8)
            fname = f"frame_{current_frame_counter:06d}.png"
            img_pil.save(os.path.join(frames_dir, fname), compress_level=1)
            current_frame_counter += 1

        with open(manifest_path, "w") as f:
            json.dump({
                "total_saved_frames": current_frame_counter,
                "last_segment_saved": segment_index
            }, f)

        final_video_path = ""
        status = f"Saved segment {segment_index} ({keep_frames.shape[0]} frames). Total frames on disk: {current_frame_counter}."
        print(f"[SCAIL_DiskVideoCollector] {status}")

        should_compile = is_last_segment or save_video_every_segment
        if should_compile and current_frame_counter > 0:
            final_video_path = os.path.join(base_dir, output_filename)
            input_pattern = os.path.join(frames_dir, "frame_%06d.png")
            ffmpeg_exe = get_ffmpeg_path()

            ffmpeg_cmd = [
                ffmpeg_exe, "-y",
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p"
            ]

            temp_audio_path = os.path.join(cache_dir, "audio.wav")
            has_audio = False
            if audio is not None and isinstance(audio, dict) and "waveform" in audio:
                try:
                    import torchaudio
                    torchaudio.save(
                        temp_audio_path,
                        audio["waveform"].squeeze(0).cpu(),
                        audio.get("sample_rate", 44100)
                    )
                    ffmpeg_cmd.extend(["-i", temp_audio_path, "-c:a", "aac", "-b:a", "192k", "-shortest"])
                    has_audio = True
                except Exception as e:
                    print(f"[SCAIL_DiskVideoCollector] Audio export warning: {e}")

            ffmpeg_cmd.append(final_video_path)

            try:
                proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    status = f"SUCCESS! Stitched {current_frame_counter} frames to {final_video_path}"
                    print(f"[SCAIL_DiskVideoCollector] {status}")
                    if is_last_segment and cleanup_temp_frames and os.path.exists(frames_dir):
                        try:
                            shutil.rmtree(frames_dir)
                            status += " (Cleaned temporary frames)"
                        except Exception:
                            pass
                else:
                    status = f"FFmpeg error: {proc.stderr[:200]}"
                    print(f"[SCAIL_DiskVideoCollector] {status}")
            except Exception as e:
                status = f"Failed to execute FFmpeg ({ffmpeg_exe}): {e}"
                print(f"[SCAIL_DiskVideoCollector] {status}")

        cached_tensor = numpy_to_tensor(overlap_frames)
        return (cached_tensor, final_video_path, status)


# =====================================================================
# Node 3: SCAIL_ReinhardColorTransfer
# =====================================================================
class SCAIL_ReinhardColorTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE", {}),
                "image_ref": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_transfer"
    CATEGORY = "SCAIL_Streamer"

    def apply_transfer(self, image_target, image_ref):
        target_np = tensor_to_numpy(image_target)
        ref_np = tensor_to_numpy(image_ref)

        ref_frame = ref_np[-1]
        res_list = []
        for i in range(target_np.shape[0]):
            res_list.append(reinhard_color_transfer_np(target_np[i], ref_frame))

        out_np = np.stack(res_list, axis=0)
        return (numpy_to_tensor(out_np),)


NODE_CLASS_MAPPINGS = {
    "SCAIL_VideoChunkIterator": SCAIL_VideoChunkIterator,
    "SCAIL_DiskVideoCollector": SCAIL_DiskVideoCollector,
    "SCAIL_ReinhardColorTransfer": SCAIL_ReinhardColorTransfer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL_VideoChunkIterator": "🎬 SCAIL Video Chunk Iterator (Long Video)",
    "SCAIL_DiskVideoCollector": "💾 SCAIL Disk Video Collector & Stitched Saver",
    "SCAIL_ReinhardColorTransfer": "🎨 SCAIL Reinhard Color Transfer"
}