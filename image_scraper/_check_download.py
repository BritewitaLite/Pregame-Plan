import os, cv2, sys
p=r"C:\Users\bryce\AppData\Local\Temp\yt_scrape_od8cia1f\scrape_tmp_video.mp4"
if not os.path.exists(p):
    print("MISSING")
    sys.exit(0)
size=os.path.getsize(p)
cap=cv2.VideoCapture(p)
fps=cap.get(cv2.CAP_PROP_FPS) or 0
frames=cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
duration=frames/fps if fps else 0
print(f"exists:1 size_bytes:{size} fps:{fps} frames:{frames} duration_s:{duration}")
cap.release()
