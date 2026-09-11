import yt_dlp, json, sys
url = "https://www.youtube.com/watch?v=4I_ak7-ruV4"
with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    print(json.dumps({'duration': info.get('duration'), 'filesize': info.get('filesize'), 'format': info.get('format')}))
