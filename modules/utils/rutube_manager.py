"""
Rutube video metadata and audio extraction via yt-dlp.
"""
import os
import subprocess
import tempfile

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def _ydl_extract_info(url: str, download: bool = False, outtmpl: str = None):
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is required for Rutube. Install with: pip install yt-dlp"
        )
    opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
    if download:
        opts["format"] = "bestaudio/best"
        opts["outtmpl"] = outtmpl or os.path.join(tempfile.gettempdir(), "rutube_%(id)s.%(ext)s")
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=download)


def get_rutube_metas(link: str):
    """
    Return (thumbnail_url, title, description) for a Rutube video.
    """
    info = _ydl_extract_info(link, download=False)
    if not info:
        return None, "", ""
    thumbnail = info.get("thumbnail") or ""
    title = info.get("title") or ""
    description = info.get("description") or ""
    return thumbnail, title, description


def get_rutube_audio(link: str) -> str:
    """
    Download audio from Rutube video and return path to temp wav file.
    Caller should delete the file when done.
    """
    tmp_dir = tempfile.mkdtemp(prefix="rutube_")
    outtmpl = os.path.join(tmp_dir, "audio.%(ext)s")
    info = _ydl_extract_info(link, download=True, outtmpl=outtmpl)
    if not info:
        return None
    # Find downloaded file (yt-dlp may use different extension)
    downloaded_path = None
    for name in os.listdir(tmp_dir):
        p = os.path.join(tmp_dir, name)
        if os.path.isfile(p):
            downloaded_path = p
            break
    if not downloaded_path:
        return None
    wav_path = os.path.join(tmp_dir, "audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", downloaded_path, "-acodec", "pcm_s16le", "-ar", "16000", wav_path],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return downloaded_path
    try:
        os.remove(downloaded_path)
    except OSError:
        pass
    return wav_path
