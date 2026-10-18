
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
        def get_input_directory(self):
            return os.path.join(os.path.dirname(self.get_output_directory()), "input")
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
    for p in ["ffmpeg.exe", "python_embeded/Scripts/ffmpeg.exe", "bin/ffmpeg.exe"]:
        fp = os.path.join(os.getcwd(), p)
        if os.path.exists(fp):
            return fp
    return "ffmpeg"

def resolve_project_path(rel_or_abs_path):
    clean = (rel_or_abs_path or "scail_project").strip()
    base = os.path.normpath(clean) if os.path.isabs(clean) else os.path.normpath(os.path.join(folder_paths.get_output_directory(), clean))
    os.makedirs(base, exist_ok=True)
    return base

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
    try:
        total_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        if torch is not None and isinstance(images, torch.Tensor):
            s1 = images[0, ::8, ::8, 0].cpu().numpy().tobytes()
            s2 = images[-1, ::8, ::8, 0].cpu().numpy().tobytes()
        else:
            s1 = images[0, ::8, ::8, 0].tobytes()
            s2 = images[-1, ::8, ::8, 0].tobytes()
        h_str = hashlib.md5(s1 + s2).hexdigest()[:8]
        return f"{total_frames}f_{w}x{h}_{h_str}"
    except Exception:
        return "legacy_fingerprint"

def reinhard_color_transfer_np(target_rgb, ref_rgb):
    try:
        target_u8 = (target_rgb * 255.0).clip(0, 255).astype(np.uint8)
        ref_u8 = (ref_rgb * 255.0).clip(0, 255).astype(np.uint8)
        target_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        res_lab = np.zeros_like(target_lab)
        for c in range(3):
            t_mean, t_std = np.mean(target_lab[:, :, c]), np.std(target_lab[:, :, c])
            r_mean, r_std = np.mean(ref_lab[:, :, c]), np.std(ref_lab[:, :, c])
            res_lab[:, :, c] = target_lab[:, :, c] if t_std < 1e-6 else ((target_lab[:, :, c] - t_mean) * (r_std / t_std)) + r_mean
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    except Exception as e:
        print(f"Color transfer failed: {e}")
        return target_rgb

def compile_frames_to_video(frames_dir, output_mp4, fps=24.0, audio_path=None):
    frames_dir_clean = os.path.abspath(frames_dir).replace("\\", "/")
    output_mp4_clean = os.path.abspath(output_mp4).replace("\\", "/")
    input_pattern = f"{frames_dir_clean}/frame_%06d.png"
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [ffmpeg_exe, "-y", "-framerate", str(fps), "-i", input_pattern]
    has_audio = audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024
    if has_audio:
        cmd.extend(["-i", os.path.abspath(audio_path).replace("\\", "/")])
    cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(output_mp4_clean)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if proc.returncode == 0 and os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 1000:
            return True, f"SUCCESS! {output_mp4} ({'with audio' if has_audio else 'no audio'})"
    except Exception as e:
        print(f"FFmpeg error: {e}")
    # fallback imageio
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not files:
        return False, "No frames to compile"
    try:
        import imageio
        with imageio.get_writer(output_mp4, fps=float(fps)) as w:
            for f in files:
                w.append_data(imageio.imread(f))
        return True, f"Stitched via imageio to {output_mp4}"
    except Exception as e:
        return False, f"Compile failed: {e}"

def load_frames_from_video_path(video_path, start_frame, frame_length):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(frame_length):
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
    cap.release()
    if not frames:
        raise ValueError(f"No frames at {start_frame} from {video_path}")
    while len(frames) < frame_length:
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0), total if total>0 else start_frame+len(frames), w, h

def resolve_video_path(video_path_str):
    """Resolve video_path to absolute file, searching input folder if needed"""
    if not video_path_str or not isinstance(video_path_str, str):
        return None
    vp = video_path_str.strip()
    if not vp:
        return None
    if os.path.isabs(vp) and os.path.exists(vp):
        return vp
    # Try relative to input folder
    try:
        input_dir = folder_paths.get_input_directory() if hasattr(folder_paths, 'get_input_directory') else os.path.join(os.path.dirname(folder_paths.get_output_directory()), "input")
    except:
        input_dir = os.path.join(os.getcwd(), "input")
    candidates = [
        vp,
        os.path.join(input_dir, vp),
        os.path.join(input_dir, os.path.basename(vp)),
        os.path.join(os.getcwd(), vp),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

class SCAIL_VideoChunkIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": "scail_project"}),
                "frame_length": ("INT", {"default": 81, "min": 16, "max": 1024}),
                "overlap_count": ("INT", {"default": 5, "min": 1, "max": 32}),
                "mode": (["Manual Segment Index", "Auto Queue Loop", "Reset Session State"], {"default": "Manual Segment Index"}),
                "manual_segment_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "Anjali Arora ❤️ Kacha badam ❤️.mp4", "multiline": False}),
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "audio": ("AUDIO",),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "BOOLEAN", "INT", "INT", "INT", "STRING", "FLOAT", "AUDIO")
    RETURN_NAMES = ("pose_chunk", "previous_frames", "segment_index", "total_segments", "is_last_segment", "video_frame_offset", "width_32", "height_32", "output_folder", "fps", "audio")
    FUNCTION = "iterate_chunk"
    CATEGORY = "SCAIL_Streamer"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def iterate_chunk(self, output_folder, frame_length, overlap_count, mode, manual_segment_index, video_path="", images=None, fps=24.0, audio=None):
        base_dir = resolve_project_path(output_folder)
        cache_dir = os.path.join(base_dir, "cache")
        frames_dir = os.path.join(base_dir, "temp_frames")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        state_file = os.path.join(cache_dir, "iterator_state.json")

        # Resolve disk video path - PRIORITY for disk-only mode
        resolved_path = resolve_video_path(video_path)
        use_disk = resolved_path is not None

        # Fingerprint
        if use_disk:
            current_fingerprint = get_video_fingerprint_from_path(resolved_path)
        elif images is not None:
            current_fingerprint = get_video_fingerprint(images)
        else:
            raise ValueError(f"SCAIL_VideoChunkIterator: No video found. video_path='{video_path}' resolved='{resolved_path}' (exists={os.path.exists(resolved_path) if resolved_path else False}). Put video in ComfyUI/input folder or provide absolute path. Also tried images input but it was None.")

        # Load state
        stored_state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    stored_state = json.load(f)
            except:
                stored_state = {}
        stored_fp = stored_state.get("video_fingerprint")
        is_new_video = (stored_fp is not None and stored_fp != current_fingerprint)

        # Reset only on explicit mode or new video - FIXES data loss bug
        if mode == "Reset Session State" or is_new_video:
            print(f"[Iterator] Resetting session (mode={mode}, new_video={is_new_video})")
            current_index = 0
            with open(state_file, "w") as f:
                json.dump({"current_index": 0, "total_segments": 0, "video_fingerprint": current_fingerprint, "last_completed": -1}, f)
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)
            for cf in ["previous_frames.npy", "last_exit_frame.npy", "manifest.json", "audio.wav", "current_processing.json"]:
                p = os.path.join(cache_dir, cf)
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
        elif mode == "Manual Segment Index":
            current_index = int(manual_segment_index)
        else: # Auto Queue Loop
            current_index = int(stored_state.get("current_index", 0))

        # Probe total frames
        if use_disk:
            cap = cv2.VideoCapture(resolved_path)
            probed_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            total_frames_val = probed_total if probed_total>0 else 1000
            orig_w, orig_h = probed_w, probed_h
        else:
            total_frames_val = images.shape[0]
            orig_h = images.shape[1]
            orig_w = images.shape[2]

        stride = frame_length - overlap_count
        if stride <= 0:
            raise ValueError(f"frame_length {frame_length} must be > overlap {overlap_count}")
        total_segments = 1 if total_frames_val <= frame_length else 1 + math.ceil((total_frames_val - frame_length)/stride)
        if current_index >= total_segments:
            current_index = total_segments - 1

        is_last = (current_index >= total_segments - 1)
        offset = current_index * stride

        width_32 = (orig_w // 32) * 32
        height_32 = (orig_h // 32) * 32

        # Load ONLY needed chunk
        if use_disk:
            pose_chunk_np, _, _, _ = load_frames_from_video_path(resolved_path, offset, frame_length)
        else:
            # Legacy: slice from full tensor (still VRAM but only 81 frames returned)
            s = offset
            e = s + frame_length
            if e <= total_frames_val:
                pose_chunk_np = tensor_to_numpy(images[s:e])
            else:
                avail = tensor_to_numpy(images[s:total_frames_val])
                miss = frame_length - avail.shape[0]
                if miss>0:
                    pad = np.repeat(avail[-1:], miss, axis=0)
                    pose_chunk_np = np.concatenate([avail, pad], axis=0)
                else:
                    pose_chunk_np = avail

        # Load previous generated overlap
        cache_prev = os.path.join(cache_dir, "previous_frames.npy")
        if current_index == 0 or not os.path.exists(cache_prev):
            prev_np = np.zeros((1, pose_chunk_np.shape[1], pose_chunk_np.shape[2], 3), dtype=np.float32)
        else:
            prev_np = np.load(cache_prev)

        # Mark current processing (do NOT advance state yet - collector will)
        with open(os.path.join(cache_dir, "current_processing.json"), "w") as f:
            json.dump({"processing_index": current_index, "video_fingerprint": current_fingerprint}, f)

        print(f"\n🎬 [SCAIL] Segment {current_index+1}/{total_segments} disk={use_disk} offset={offset} last={is_last}\n")

        return (numpy_to_tensor(pose_chunk_np), numpy_to_tensor(prev_np), current_index, total_segments, is_last, offset, width_32, height_32, output_folder, fps, audio)

class SCAIL_DiskVideoCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_images": ("IMAGE", {}),
                "output_folder": ("STRING", {"default": "scail_project"}),
                "segment_index": ("INT", {"default": 0}),
                "is_last_segment": ("BOOLEAN", {"default": False}),
                "overlap_count": ("INT", {"default": 5}),
                "apply_color_transfer": ("BOOLEAN", {"default": True}),
                "fps": ("FLOAT", {"default": 24.0}),
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
        if not output_filename.lower().endswith(".mp4"):
            output_filename = "final_scail_video.mp4"
        try: fps = float(fps) if float(fps)>0 else 24.0
        except: fps = 24.0

        ref_path = os.path.join(cache_dir, "last_exit_frame.npy")
        prev_path = os.path.join(cache_dir, "previous_frames.npy")
        manifest_path = os.path.join(cache_dir, "manifest.json")
        state_file = os.path.join(cache_dir, "iterator_state.json")

        img_np = tensor_to_numpy(generated_images)
        final_video_path = os.path.join(base_dir, output_filename)

        # Resume-safe counter
        if segment_index == 0:
            try:
                with open(state_file, "r") as f:
                    st = json.load(f)
                last_completed = st.get("last_completed", -1)
            except:
                last_completed = -1
            should_clean = (last_completed == -1) or (not os.path.exists(manifest_path))
            if should_clean:
                current_counter = 0
                for ef in os.listdir(frames_dir):
                    if ef.endswith(".png"):
                        try: os.remove(os.path.join(frames_dir, ef))
                        except: pass
            else:
                try:
                    with open(manifest_path, "r") as f:
                        m = json.load(f)
                    current_counter = m.get("total_saved_frames", 0)
                except:
                    current_counter = 0
        else:
            try:
                with open(manifest_path, "r") as f:
                    m = json.load(f)
                current_counter = m.get("total_saved_frames", 0)
            except:
                current_counter = 0

        keep = img_np if segment_index==0 else img_np[overlap_count:]
        if segment_index!=0 and apply_color_transfer and os.path.exists(ref_path):
            try:
                ref = np.load(ref_path)
                keep = np.stack([reinhard_color_transfer_np(keep[i], ref) for i in range(keep.shape[0])], axis=0)
            except Exception as e:
                print(f"Color transfer warning: {e}")

        np.save(ref_path, img_np[-1])
        overlap_frames = img_np[-overlap_count:]
        np.save(prev_path, overlap_frames)

        for i in range(keep.shape[0]):
            u8 = (keep[i]*255.0).clip(0,255).astype(np.uint8)
            Image.fromarray(u8).save(os.path.join(frames_dir, f"frame_{current_counter:06d}.png"), compress_level=1)
            current_counter+=1

        with open(manifest_path, "w") as f:
            json.dump({"total_saved_frames": current_counter, "last_segment_saved": segment_index}, f)

        temp_audio = os.path.join(cache_dir, "audio.wav")
        if audio and isinstance(audio, dict) and "waveform" in audio:
            try:
                wf = audio["waveform"]
                if wf.numel()>0:
                    import torchaudio
                    torchaudio.save(temp_audio, wf.squeeze(0).cpu(), audio.get("sample_rate", 44100))
            except Exception as e:
                print(f"Audio save warning: {e}")

        if current_counter>0:
            ok, comp_status = compile_frames_to_video(frames_dir, final_video_path, fps=fps, audio_path=temp_audio if os.path.exists(temp_audio) else None)
            status = comp_status
            if ok and is_last_segment and cleanup_temp_frames and os.path.exists(frames_dir):
                try: shutil.rmtree(frames_dir); status+=" (Cleaned)"
                except: pass
        else:
            status = f"Saved segment {segment_index}"

        # Advance state ONLY after success - FIXES abort bug
        try:
            st = {}
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    st = json.load(f)
            fp = st.get("video_fingerprint","")
            next_idx = 0 if is_last_segment else segment_index+1
            with open(state_file, "w") as f:
                json.dump({"current_index": next_idx, "total_segments": st.get("total_segments",0), "video_fingerprint": fp, "last_completed": segment_index}, f)
            cp = os.path.join(cache_dir, "current_processing.json")
            if os.path.exists(cp):
                os.remove(cp)
        except Exception as e:
            print(f"State advance warning: {e}")

        cached_tensor = numpy_to_tensor(overlap_frames)
        ui_payload = {}
        if os.path.exists(final_video_path) and os.path.getsize(final_video_path)>1000:
            rel = os.path.relpath(base_dir, folder_paths.get_output_directory()).replace("\\","/")
            if rel==".": rel=""
            ui_payload["gifs"] = [{"filename": output_filename, "subfolder": rel, "type": "output", "format": "video/mp4"}]

        return {"ui": ui_payload, "result": (cached_tensor, final_video_path, status)}

class SCAIL_ReinhardColorTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_target": ("IMAGE", {}), "image_ref": ("IMAGE", {})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_transfer"
    CATEGORY = "SCAIL_Streamer"
    def apply_transfer(self, image_target, image_ref):
        target_np = tensor_to_numpy(image_target)
        ref_np = tensor_to_numpy(image_ref)
        ref_frame = ref_np[-1]
        out_np = np.stack([reinhard_color_transfer_np(target_np[i], ref_frame) for i in range(target_np.shape[0])], axis=0)
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
