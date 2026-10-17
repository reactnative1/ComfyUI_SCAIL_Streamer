import os
import json
import math
import glob
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


def compile_frames_to_video(frames_dir, output_mp4, fps=24.0, audio_path=None):
    frames_dir_clean = os.path.abspath(frames_dir).replace("\\", "/")
    output_mp4_clean = os.path.abspath(output_mp4).replace("\\", "/")
    input_pattern = f"{frames_dir_clean}/frame_%06d.png"
    ffmpeg_exe = get_ffmpeg_path()

    # 1. Start with global options
    ffmpeg_cmd = [ffmpeg_exe, "-y"]

    # 2. Add video frames input
    ffmpeg_cmd.extend(["-framerate", str(fps), "-i", input_pattern])

    # 3. Add audio input if valid (MUST come before output options)
    has_audio = False
    if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
        audio_clean = os.path.abspath(audio_path).replace("\\", "/")
        ffmpeg_cmd.extend(["-i", audio_clean])
        has_audio = True

    # 4. Output encoding options (strictly after all -i inputs)
    ffmpeg_cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    ])

    if has_audio:
        ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    # 5. Output file
    ffmpeg_cmd.append(output_mp4_clean)

    try:
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 1000:
            msg = f"SUCCESS! Stitched via FFmpeg to {output_mp4}" + (" (with audio)" if has_audio else "")
            print(f"[SCAIL_DiskVideoCollector] {msg}")
            return True, msg
        else:
            err_lines = proc.stderr.strip().splitlines()[-3:] if proc.stderr else []
            err_detail = " | ".join(err_lines)
            print(f"[SCAIL_DiskVideoCollector] FFmpeg error: {err_detail}. Falling back to imageio...")
    except Exception as e:
        print(f"[SCAIL_DiskVideoCollector] FFmpeg execution issue: {e}. Falling back to imageio...")

    # Fallback to imageio
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        return False, "No frame PNG files found in temp_frames to compile."

    try:
        import imageio
        with imageio.get_writer(output_mp4, fps=float(fps)) as writer:
            for ff in frame_files:
                writer.append_data(imageio.imread(ff))
        if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 1000:
            msg = f"SUCCESS! Stitched via imageio fallback to {output_mp4}"
            print(f"[SCAIL_DiskVideoCollector] {msg}")
            return True, msg
    except Exception as e:
        print(f"[SCAIL_DiskVideoCollector] imageio fallback also failed: {e}")

    return False, "Failed to compile video with both FFmpeg and imageio."


# =====================================================================
# Node 1: SCAIL_VideoChunkIterator
# =====================================================================
class SCAIL_VideoChunkIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The complete source video frames from LoadVideo / GetVideoComponents."}),
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
            for cache_f in ["previous_frames.npy", "last_exit_frame.npy", "manifest.json", "audio.wav"]:
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

        print(f"[SCAIL_VideoChunkIterator] Processing Segment {current_index + 1}/{total_segments} (Frames {start_frame}..{end_frame}). Is Last: {is_last_segment}")

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
                "save_video_every_segment": ("BOOLEAN", {"default": True, "tooltip": "Compile MP4 after every segment so you can preview progress immediately."}),
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

        final_video_path = os.path.join(base_dir, output_filename)
        status = f"Saved segment {segment_index} ({keep_frames.shape[0]} frames). Total frames on disk: {current_frame_counter}."
        print(f"[SCAIL_DiskVideoCollector] {status}")

        temp_audio_path = os.path.join(cache_dir, "audio.wav")
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            try:
                wf = audio["waveform"]
                if wf.numel() > 0:
                    import torchaudio
                    torchaudio.save(
                        temp_audio_path,
                        wf.squeeze(0).cpu(),
                        audio.get("sample_rate", 44100)
                    )
            except Exception as e:
                print(f"[SCAIL_DiskVideoCollector] Audio save warning: {e}")

        should_compile = is_last_segment or save_video_every_segment
        if should_compile and current_frame_counter > 0:
            ok, comp_status = compile_frames_to_video(
                frames_dir=frames_dir,
                output_mp4=final_video_path,
                fps=fps,
                audio_path=temp_audio_path if os.path.exists(temp_audio_path) else None
            )
            status = comp_status
            if ok and is_last_segment and cleanup_temp_frames and os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                    status += " (Cleaned temporary frames)"
                except Exception:
                    pass

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