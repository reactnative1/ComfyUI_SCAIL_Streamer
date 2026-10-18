
import os
import json
import math
import glob
import hashlib
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


def get_video_fingerprint_from_path(video_path):
    try:
        st = os.stat(video_path)
        return f"{os.path.basename(video_path)}_{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        return video_path


def get_video_fingerprint(images):
    # Legacy fallback if images tensor is passed
    total_frames = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    if torch is not None and isinstance(images, torch.Tensor):
        s1 = images[0, ::16, ::16, 0].cpu().numpy().tobytes()
        s2 = images[-1, ::16, ::16, 0].cpu().numpy().tobytes()
    else:
        s1 = images[0, ::16, ::16, 0].tobytes()
        s2 = images[-1, ::16, ::16, 0].tobytes()
    h_str = hashlib.md5(s1 + s2).hexdigest()[:8]
    return f"{total_frames}f_{w}x{h}_{h_str}"


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
    ffmpeg_cmd = [ffmpeg_exe, "-y"]
    ffmpeg_cmd.extend(["-framerate", str(fps), "-i", input_pattern])
    has_audio = False
    if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
        audio_clean = os.path.abspath(audio_path).replace("\\", "/")
        ffmpeg_cmd.extend(["-i", audio_clean])
        has_audio = True
    ffmpeg_cmd.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    ])
    if has_audio:
        ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
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


def load_frames_from_video_path(video_path, start_frame, frame_length, total_frames_hint=None):
    """Disk-only loader: reads only needed frames via cv2, returns numpy array float32 0-1 and meta."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video_path {video_path}")
    # Try to get total frames if not provided
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 and total_frames_hint:
        total_frames = total_frames_hint
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for i in range(frame_length):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path} at {start_frame}")
    # Pad if needed by repeating last
    if len(frames) < frame_length:
        last = frames[-1]
        while len(frames) < frame_length:
            frames.append(last.copy())
    arr = np.stack(frames, axis=0)  # [T,H,W,3]
    return arr, total_frames if total_frames>0 else (start_frame + len(frames)), width, height


# =====================================================================
# Node 1: SCAIL_VideoChunkIterator - DISK ONLY + RESUME SAFE
# =====================================================================
class SCAIL_VideoChunkIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": "scail_project", "multiline": False}),
                "frame_length": ("INT", {"default": 81, "min": 16, "max": 1024, "step": 1}),
                "overlap_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),
                "mode": (["Auto Queue Loop", "Manual Segment Index", "Reset Session State"], {"default": "Manual Segment Index"}),
                "manual_segment_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "LEGACY: full video tensor. If provided, uses it. Disk-only mode uses video_path."}),
                "video_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Disk-only: absolute path to driving video file. If set, ignores images tensor."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "audio": ("AUDIO", {}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "Optional override for total frames when using video_path without reliable cap"}),
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

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def iterate_chunk(self, output_folder, frame_length, overlap_count, mode, manual_segment_index, images=None, video_path="", fps=24.0, audio=None, total_frames=0):
        # Determine source
        use_disk_path = isinstance(video_path, str) and video_path.strip() != "" and os.path.exists(video_path.strip())
        base_dir = resolve_project_path(output_folder)
        cache_dir = os.path.join(base_dir, "cache")
        frames_dir = os.path.join(base_dir, "temp_frames")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        state_file = os.path.join(cache_dir, "iterator_state.json")
        # Fingerprint
        if use_disk_path:
            current_fingerprint = get_video_fingerprint_from_path(video_path.strip())
        elif images is not None:
            current_fingerprint = get_video_fingerprint(images)
        else:
            raise ValueError("SCAIL_VideoChunkIterator: Need either video_path or images")

        stored_state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    stored_state = json.load(f)
            except Exception:
                stored_state = {}

        stored_fingerprint = stored_state.get("video_fingerprint")
        is_new_video = (stored_fingerprint is not None and stored_fingerprint != current_fingerprint)

        # Reset handling - ONLY place where we delete frames
        if mode == "Reset Session State" or is_new_video:
            if is_new_video:
                print(f"[SCAIL_VideoChunkIterator] New video detected! Resetting {output_folder}")
            current_index = 0
            # Write fresh state with current_index = 0, do NOT delete here? Delete only on explicit reset or new video
            with open(state_file, "w") as f:
                json.dump({"current_index": 0, "total_segments": 0, "video_fingerprint": current_fingerprint, "last_completed": -1}, f)
            # Clean frames only on reset/new video
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)
            for cache_f in ["previous_frames.npy", "last_exit_frame.npy", "manifest.json", "audio.wav", "current_processing.json"]:
                cp = os.path.join(cache_dir, cache_f)
                if os.path.exists(cp):
                    try: os.remove(cp)
                    except: pass
        elif mode == "Manual Segment Index":
            current_index = manual_segment_index
        else: # Auto Queue Loop
            current_index = stored_state.get("current_index", 0)

        # Determine total_frames, width, height without loading full video into memory if using disk path
        if use_disk_path:
            # Probe total frames quickly
            cap = cv2.VideoCapture(video_path.strip())
            probed_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if total_frames and total_frames > 0:
                total_frames_val = total_frames
            elif probed_total > 0:
                total_frames_val = probed_total
            else:
                total_frames_val = 999999  # unknown, will be handled by padding
            orig_width, orig_height = probed_w, probed_h
        else:
            total_frames_val = images.shape[0]
            orig_height = images.shape[1]
            orig_width = images.shape[2]

        stride = frame_length - overlap_count
        if stride <= 0:
            raise ValueError(f"frame_length ({frame_length}) must be > overlap_count ({overlap_count})")
        if total_frames_val <= frame_length:
            total_segments = 1
        else:
            total_segments = 1 + math.ceil((total_frames_val - frame_length) / stride)

        if current_index >= total_segments:
            current_index = total_segments - 1

        is_last_segment = (current_index >= total_segments - 1)
        video_frame_offset = current_index * stride
        start_frame = current_index * stride
        end_frame = start_frame + frame_length

        width_32 = (orig_width // 32) * 32
        height_32 = (orig_height // 32) * 32

        # Load only needed chunk
        if use_disk_path:
            pose_chunk_np, _, _, _ = load_frames_from_video_path(video_path.strip(), start_frame, frame_length, total_frames_val)
        else:
            # Legacy tensor slice - still only 81 frames in memory
            if end_frame <= total_frames_val:
                pose_chunk_np = tensor_to_numpy(images[start_frame:end_frame])
            else:
                available = tensor_to_numpy(images[start_frame:total_frames_val])
                missing = frame_length - available.shape[0]
                if missing > 0:
                    pad = np.repeat(available[-1:], missing, axis=0)
                    pose_chunk_np = np.concatenate([available, pad], axis=0)
                else:
                    pose_chunk_np = available

        # previous_frames from disk cache (generated overlap)
        cache_prev_file = os.path.join(cache_dir, "previous_frames.npy")
        if current_index == 0 or not os.path.exists(cache_prev_file):
            previous_frames = np.zeros((1, pose_chunk_np.shape[1], pose_chunk_np.shape[2], 3), dtype=np.float32)
        else:
            cached_arr = np.load(cache_prev_file)
            previous_frames = cached_arr

        # DO NOT advance state here. Write current_processing marker. Collector will advance after success.
        with open(os.path.join(cache_dir, "current_processing.json"), "w") as f:
            json.dump({"processing_index": current_index, "video_fingerprint": current_fingerprint}, f)

        print(f"\n{'='*55}")
        print(f"🎬 [SCAIL Streamer] Segment {current_index + 1} / {total_segments} (disk-only)")
        print(f"   Frames: {start_frame} to {end_frame} | Is Last: {is_last_segment}")
        print(f"{'='*55}\n")

        return (
            numpy_to_tensor(pose_chunk_np), numpy_to_tensor(previous_frames), current_index, total_segments,
            is_last_segment, video_frame_offset, width_32, height_32,
            output_folder, fps, audio
        )


# =====================================================================
# Node 2: SCAIL_DiskVideoCollector - RESUME SAFE
# =====================================================================
class SCAIL_DiskVideoCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_images": ("IMAGE", {}),
                "output_folder": ("STRING", {"default": "scail_project"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "is_last_segment": ("BOOLEAN", {"default": False}),
                "overlap_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),
                "apply_color_transfer": ("BOOLEAN", {"default": True}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "output_filename": ("STRING", {"default": "final_scail_video.mp4"}),
                "cleanup_temp_frames": ("BOOLEAN", {"default": False}),
            },
            "optional": {"audio": ("AUDIO", {})}
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("cached_last_frames", "final_video_path", "status")
    FUNCTION = "collect_and_save"
    CATEGORY = "SCAIL_Streamer"
    OUTPUT_NODE = True

    def collect_and_save(self, generated_images, output_folder, segment_index, is_last_segment, overlap_count, apply_color_transfer, fps=24.0, output_filename="final_scail_video.mp4", cleanup_temp_frames=False, audio=None, **kwargs):
        base_dir = resolve_project_path(output_folder)
        frames_dir = os.path.join(base_dir, "temp_frames")
        cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        if not isinstance(output_filename, str) or not output_filename.lower().endswith(".mp4"):
            output_filename = "final_scail_video.mp4"
        try:
            fps = float(fps)
            if fps <= 0: fps = 24.0
        except: fps = 24.0

        ref_frame_path = os.path.join(cache_dir, "last_exit_frame.npy")
        prev_frames_path = os.path.join(cache_dir, "previous_frames.npy")
        manifest_path = os.path.join(cache_dir, "manifest.json")
        state_file = os.path.join(cache_dir, "iterator_state.json")

        img_np = tensor_to_numpy(generated_images)
        final_video_path = os.path.join(base_dir, output_filename)

        # RESUME SAFE: Only delete frames if segment_index==0 AND manifest does NOT exist or is explicitly reset via iterator
        # Do NOT delete if resuming same video at segment 0 after failure - check if manifest says last_saved == -1
        if segment_index == 0:
            # If manifest exists and total_saved_frames>0, we are resuming? Actually for true restart we want delete.
            # We use iterator_state.json last_completed to decide: if last_completed == -1, it's fresh start -> delete.
            try:
                with open(state_file, "r") as f:
                    st = json.load(f)
                last_completed = st.get("last_completed", -1)
            except:
                last_completed = -1
            # Only delete if fresh start (last_completed == -1) or no manifest
            should_clean = (last_completed == -1) or (not os.path.exists(manifest_path))
            if should_clean:
                if os.path.exists(final_video_path):
                    backup_path = os.path.join(base_dir, f"{os.path.splitext(output_filename)[0]}_prev.mp4")
                    try: shutil.copy2(final_video_path, backup_path)
                    except: pass
                current_frame_counter = 0
                for existing_f in os.listdir(frames_dir):
                    if existing_f.endswith(".png"):
                        try: os.remove(os.path.join(frames_dir, existing_f))
                        except: pass
            else:
                # Resuming at 0 after failure - keep existing frames, load counter
                try:
                    with open(manifest_path, "r") as f:
                        m = json.load(f)
                    current_frame_counter = m.get("total_saved_frames", 0)
                except:
                    current_frame_counter = 0
        else:
            try:
                with open(manifest_path, "r") as f:
                    m = json.load(f)
                current_frame_counter = m.get("total_saved_frames", 0)
            except:
                current_frame_counter = 0

        if segment_index == 0:
            # For first segment, we already handled counter above, but ensure keep all frames
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
            json.dump({"total_saved_frames": current_frame_counter, "last_segment_saved": segment_index}, f)

        status = f"Saved segment {segment_index} ({keep_frames.shape[0]} frames). Total: {current_frame_counter}."

        temp_audio_path = os.path.join(cache_dir, "audio.wav")
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            try:
                wf = audio["waveform"]
                if wf.numel() > 0:
                    import torchaudio
                    torchaudio.save(temp_audio_path, wf.squeeze(0).cpu(), audio.get("sample_rate", 44100))
            except Exception as e:
                print(f"[SCAIL_DiskVideoCollector] Audio save warning: {e}")

        if current_frame_counter > 0:
            ok, comp_status = compile_frames_to_video(frames_dir, final_video_path, fps=fps, audio_path=temp_audio_path if os.path.exists(temp_audio_path) else None)
            status = comp_status
            if ok and is_last_segment and cleanup_temp_frames and os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                    status += " (Cleaned temp frames)"
                except: pass

        # SUCCESS: Now advance iterator state
        try:
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    st = json.load(f)
            else:
                st = {}
            fingerprint = st.get("video_fingerprint", "")
            next_index = segment_index + 1
            if is_last_segment:
                next_index = 0  # wrap for next full video
                last_completed = segment_index
            else:
                last_completed = segment_index
            with open(state_file, "w") as f:
                json.dump({"current_index": next_index, "total_segments": st.get("total_segments", 0), "video_fingerprint": fingerprint, "last_completed": last_completed}, f)
            # Remove processing marker
            proc_file = os.path.join(cache_dir, "current_processing.json")
            if os.path.exists(proc_file):
                os.remove(proc_file)
        except Exception as e:
            print(f"[SCAIL_DiskVideoCollector] State advance warning: {e}")

        cached_tensor = numpy_to_tensor(overlap_frames)
        ui_payload = {}
        if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 1000:
            rel_subfolder = os.path.relpath(base_dir, folder_paths.get_output_directory()).replace("\\", "/")
            if rel_subfolder == ".": rel_subfolder = ""
            ui_payload["gifs"] = [{"filename": output_filename, "subfolder": rel_subfolder, "type": "output", "format": "video/mp4"}]

        return {"ui": ui_payload, "result": (cached_tensor, final_video_path, status)}


# =====================================================================
# Node 3: SCAIL_ReinhardColorTransfer
# =====================================================================
class SCAIL_ReinhardColorTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_target": ("IMAGE", {}), "image_ref": ("IMAGE", {})}}
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
