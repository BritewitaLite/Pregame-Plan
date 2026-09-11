import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path
import cv2

try:
    from pytube import YouTube
except Exception:
    YouTube = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


def download_youtube(url, out_dir):
    # try pytube first
    if YouTube is not None:
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if stream is not None:
                out_path = stream.download(output_path=out_dir, filename='scrape_tmp_video.mp4')
                return out_path
        except Exception:
            pass

    # fallback to yt-dlp if available
    if yt_dlp is not None:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': os.path.join(out_dir, 'scrape_tmp_video.%(ext)s'),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get('ext', 'mp4')
            return os.path.join(out_dir, f'scrape_tmp_video.{ext}')

    raise RuntimeError('No downloader available. Install "pytube" or "yt-dlp"')


class ModelWrapper:
    def __init__(self, model_path, conf_thresh=0.25):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.backend = None
        self.model = None
        self._load()

    def _load(self):
        if YOLO is not None:
            try:
                self.model = YOLO(self.model_path)
                self.backend = 'ultralytics'
                return
            except Exception:
                self.model = None
        if torch is not None:
            try:
                # try yolov5 hub as fallback
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, trust_repo=True)
                self.backend = 'yolov5'
                return
            except Exception:
                self.model = None
        raise RuntimeError('No supported model backend available. Install "ultralytics" or have torch and internet for torch.hub.')

    def detect_presnap(self, frame, presnap_label='presnap', conf_needed=0.90):
        """Return (True, confidence) if presnap detected with >= conf_needed, else (False, 0.0)."""
        target_label = presnap_label.strip().lower()
        if self.backend == 'ultralytics':
            # ultralytics accepts BGR or RGB ndarray; prefer RGB
            img = frame[:, :, ::-1]
            results = self.model(img)
            if len(results) == 0:
                return False, 0.0
            r = results[0]
            # boxes may be empty
            boxes = getattr(r, 'boxes', None)
            names = getattr(self.model, 'names', None) or getattr(r, 'names', None) or {}
            if boxes is None or len(boxes) == 0:
                return False, 0.0
            # boxes.conf, boxes.cls available
            try:
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback: iterate
                for b in boxes:
                    try:
                        cid = int(b.cls.cpu().numpy())
                        conf = float(b.conf.cpu().numpy())
                    except Exception:
                        continue
                    name = names.get(cid, str(cid))
                    try:
                        name_str = str(name).strip().lower()
                    except Exception:
                        name_str = str(name).lower()
                    if name_str == target_label and conf >= conf_needed:
                        return True, conf
                return False, 0.0
            for cid, conf in zip(cls_ids, confs):
                name = names.get(int(cid), str(cid))
                try:
                    name_str = str(name).strip().lower()
                except Exception:
                    name_str = str(name).lower()
                if name_str == target_label and float(conf) >= conf_needed:
                    return True, float(conf)
            return False, 0.0

        if self.backend == 'yolov5':
            # yolov5 model expects numpy BGR
            results = self.model(frame)
            try:
                df = results.pandas().xyxy[0]
                namecol = df['name'].astype(str).str.strip().str.lower()
                pres = df[(namecol == target_label) & (df['confidence'] >= conf_needed)]
                if len(pres) > 0:
                    return True, float(pres['confidence'].max())
            except Exception:
                pass
            return False, 0.0

        return False, 0.0


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_frame(frame, out_dir, prefix='frame'):
    ensure_dir(out_dir)
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}.jpg"
    out_path = Path(out_dir) / fname
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def run_scraper(youtube_url, model_path=None, out_dir=None, conf_threshold=0.90, interval=0.8, duration=4.8, max_video_seconds=None):
    tmpd = tempfile.mkdtemp(prefix='yt_scrape_')
    try:
        print('Downloading video...')
        video_path = download_youtube(youtube_url, tmpd)
        print('Video downloaded to', video_path)

        model_path = model_path or os.path.join('models', 'player_detector.pt')
        out_dir = out_dir or os.path.join('image_scraper', 'photos')
        ensure_dir(out_dir)

        print('Loading model...')
        model = ModelWrapper(model_path)
        print('Model loaded (backend=%s)' % model.backend)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Could not open video file')

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        last_trigger = 0
        cooldown = duration + .1
        frame_idx = 0
        print('Scanning frames...')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # stop after max_video_seconds if requested
            if max_video_seconds is not None:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                if (pos_ms / 1000.0) >= float(max_video_seconds):
                    print(f'Reached max video seconds: {max_video_seconds}s, stopping.')
                    break
            frame_idx += 1
            now = time.time()
            if now - last_trigger < cooldown:
                # skip detection while in cooldown
                continue

            try:
                found, conf = model.detect_presnap(frame, conf_needed=conf_threshold)
            except Exception as e:
                print('Model detection error:', e)
                found = False

            if found:
                print(f'Presnap detected (conf={conf:.3f}) at frame {frame_idx}, saving screenshots...')
                # initial save
                save_frame(frame, out_dir, prefix='presnap')
                start = time.time()
                next_t = interval
                while True:
                    elapsed = time.time() - start
                    if elapsed > duration:
                        break
                    # read next frame (if available)
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    if elapsed >= next_t - 1e-3:
                        save_frame(frame2, out_dir, prefix='presnap')
                        next_t += interval
                last_trigger = time.time()

        cap.release()
        print('Done. Photos saved to', out_dir)

    finally:
        try:
            shutil.rmtree(tmpd)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Scrape YouTube video and capture screenshots on presnap detection')
    parser.add_argument('youtube_url')
    parser.add_argument('--model', default=os.path.join('models', 'player_detector.pt'))
    parser.add_argument('--out', default=os.path.join('image_scraper', 'photos'))
    parser.add_argument('--conf', type=float, default=0.90)
    parser.add_argument('--interval', type=float, default=.8, help='seconds between additional screenshots')
    parser.add_argument('--duration', type=float, default=4.8, help='seconds after initial screenshot to keep taking screenshots')
    parser.add_argument('--max-video-seconds', type=float, default=None, help='process only the first N seconds of the video')
    args = parser.parse_args()
    run_scraper(args.youtube_url, model_path=args.model, out_dir=args.out, conf_threshold=args.conf, interval=args.interval, duration=args.duration, max_video_seconds=args.max_video_seconds)


if __name__ == '__main__':
    main()
