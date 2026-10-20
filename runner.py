"""Penjaga bot: ambil bot.py & requirements.txt terbaru dari GitHub,
jalankan bot, lalu restart otomatis kalau bot berhenti atau ada update."""
import os
import subprocess
import sys
import time
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FILES = ["bot.py", "requirements.txt"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_env(path=".env"):
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


CFG = load_env()
RAW = CFG.get("REPO_RAW", "").rstrip("/")
TOKEN = CFG.get("GITHUB_TOKEN", "")
INTERVAL = int(CFG.get("CHECK_INTERVAL", "120") or 120)
if not RAW:
    sys.exit("REPO_RAW belum diisi di file .env")


def fetch(name):
    req = urllib.request.Request(f"{RAW}/{name}", headers={"User-Agent": "tele-bot-runner"})
    if TOKEN:
        req.add_header("Authorization", f"token {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def sync():
    """Unduh file dari GitHub. Mengembalikan daftar file yang berubah."""
    changed = []
    for name in FILES:
        try:
            data = fetch(name)
        except Exception as e:
            log(f"gagal mengambil {name}: {e}")
            continue
        if not data.strip():
            continue
        if name == "bot.py" and b"def main" not in data:
            log("bot.py dari GitHub tampak tidak valid, dilewati")
            continue
        old = open(name, "rb").read() if os.path.exists(name) else None
        if old != data:
            tmp = name + ".new"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, name)
            changed.append(name)
    return changed


def pip_install():
    log("memasang paket (pip)...")
    return subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]).returncode


def start_bot():
    log("menjalankan bot.py")
    return subprocess.Popen([sys.executable, "bot.py"])


def stop_bot(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    changed = sync()
    if not os.path.exists("bot.py"):
        sys.exit("bot.py tidak ada dan gagal diunduh. Cek REPO_RAW di .env")
    if "requirements.txt" in changed or not os.path.exists(".deps_ok"):
        if pip_install() == 0:
            open(".deps_ok", "w").close()
    proc = start_bot()
    last = time.time()
    while True:
        time.sleep(5)
        if proc.poll() is not None:
            log(f"bot berhenti (kode {proc.returncode}), mulai ulang 10 detik lagi")
            time.sleep(10)
            proc = start_bot()
            continue
        if time.time() - last >= INTERVAL:
            last = time.time()
            changed = sync()
            if changed:
                log(f"update dari GitHub: {', '.join(changed)}")
                if "requirements.txt" in changed:
                    pip_install()
                stop_bot(proc)
                proc = start_bot()


if __name__ == "__main__":
    main()
