#!/usr/bin/env python3
import os
import sys
import subprocess
import urllib.request
import time
import uuid
import threading
import hmac
import secrets
import stat

from rctl.config import load_server_config, get_runtime_env

try:
    import fastapi
    import uvicorn
    import nest_asyncio
    import pydantic
    from fastapi import FastAPI, UploadFile, File, Header, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "fastapi", "uvicorn", "pydantic", "nest-asyncio", "python-multipart", "requests"
    ])
    import fastapi
    import uvicorn
    import nest_asyncio
    import pydantic
    from fastapi import FastAPI, UploadFile, File, Header, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

config = load_server_config()

AUTH_TOKEN = config.get("AUTH_TOKEN")
if not AUTH_TOKEN:
    AUTH_TOKEN = secrets.token_urlsafe(32)
    print(
        "\nNo AUTH_TOKEN configured. Generated a one-time random token for "
        "this session:\n"
        f"    {AUTH_TOKEN}\n"
        "Set AUTH_TOKEN in your server config to persist a token across "
        "restarts. This server refuses to run without authentication.\n"
    )

CF_TUNNEL_TOKEN = config.get("CL_TUNNEL_TOKEN", "")

RUNNING_IN_COLAB = 'google.colab' in sys.modules
RUNNING_IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

nest_asyncio.apply()

app = FastAPI()


@app.middleware("http")
async def enforce_auth_token(request, call_next):
    supplied = request.headers.get("x-auth-token") or ""
    if not hmac.compare_digest(supplied, AUTH_TOKEN):
        return JSONResponse(status_code=401, content={"error": "Unauthorized", "exit_code": 401})

    return await call_next(request)

if RUNNING_IN_COLAB:
    SERVER_ROOT = "/content"
elif RUNNING_IN_KAGGLE:
    SERVER_ROOT = "/kaggle/working"
else:
    SERVER_ROOT = os.getcwd()

os.makedirs(SERVER_ROOT, exist_ok=True)

# Track active tasks
tasks = {}
tasks_lock = threading.Lock()
task_dir = f"{SERVER_ROOT}/tasks"
os.makedirs(task_dir, exist_ok=True)

MAX_TASK_AGE_SECONDS = 3600         # drop finished tasks after 1 hour
MAX_TRACKED_TASKS = 200             # hard cap on in-memory task records

class Task:
    def __init__(self, task_id, command, cwd):
        self.task_id = task_id
        self.command = command
        self.cwd = cwd
        self.stdout_path = os.path.join(task_dir, f"{task_id}.stdout")
        self.stderr_path = os.path.join(task_dir, f"{task_id}.stderr")
        self.stdout_file = open(self.stdout_path, "w+")
        self.stderr_file = open(self.stderr_path, "w+")

        marker = "__CWD_MARKER__"
        full_command = f"{command}\necho '{marker}'\npwd"

        self.proc = subprocess.Popen(
            full_command,
            shell=True,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            cwd=cwd,
            text=True
        )
        self.start_time = time.time()
        self.finished_time = None
        self.last_read_offset_stdout = 0
        self.last_read_offset_stderr = 0
        self.completed = False
        self.exit_code = None

    def check_status(self):
        if self.completed:
            return

        self.exit_code = self.proc.poll()
        if self.exit_code is not None:
            self.completed = True
            self.finished_time = time.time()
            self.stdout_file.close()
            self.stderr_file.close()
        elif time.time() - self.start_time > 300:  # 5 min timeout
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.exit_code = -9
            self.completed = True
            self.finished_time = time.time()
            self.stdout_file.close()
            self.stderr_file.close()

    def cleanup_files(self):
        for path in (self.stdout_path, self.stderr_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

def gc_tasks():
    """Remove old/completed tasks so state doesn't grow without bound."""
    now = time.time()
    with tasks_lock:
        stale_ids = [
            tid for tid, t in tasks.items()
            if t.completed and t.finished_time and (now - t.finished_time) > MAX_TASK_AGE_SECONDS
        ]
        for tid in stale_ids:
            tasks[tid].cleanup_files()
            del tasks[tid]

        if len(tasks) > MAX_TRACKED_TASKS:
            # Drop oldest completed tasks first
            completed_ids = sorted(
                (tid for tid, t in tasks.items() if t.completed),
                key=lambda tid: tasks[tid].finished_time or 0
            )
            overflow = len(tasks) - MAX_TRACKED_TASKS
            for tid in completed_ids[:overflow]:
                tasks[tid].cleanup_files()
                del tasks[tid]

class CodeExecutionRequest(BaseModel):
    command: str

@app.post("/exec")
async def execute_command(req: CodeExecutionRequest, x_auth_token: str = Header(None)):
    global SERVER_ROOT
    gc_tasks()
    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = Task(task_id, req.command, SERVER_ROOT)
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str, x_auth_token: str = Header(None)):
    global SERVER_ROOT
    if task_id not in tasks:
        return {"error": "Task not found", "exit_code": 404}

    task = tasks[task_id]
    task.check_status()

    # Read new stdout content
    new_stdout = ""
    if os.path.exists(task.stdout_path):
        with open(task.stdout_path, "r") as f:
            f.seek(task.last_read_offset_stdout)
            new_stdout = f.read()
            task.last_read_offset_stdout = f.tell()

    # Read new stderr content
    new_stderr = ""
    if os.path.exists(task.stderr_path):
        with open(task.stderr_path, "r") as f:
            f.seek(task.last_read_offset_stderr)
            new_stderr = f.read()
            task.last_read_offset_stderr = f.tell()

    # If completed, check for CWD update
    cwd = task.cwd
    marker = "__CWD_MARKER__"
    if task.completed:
        if os.path.exists(task.stdout_path):
            with open(task.stdout_path, "r") as f:
                full_stdout = f.read()
            if marker in full_stdout:
                parts = full_stdout.split(marker)
                new_cwd = parts[1].strip()
                if os.path.isdir(new_cwd):
                    task.cwd = new_cwd
                    cwd = new_cwd
                # Strip marker and CWD from the new_stdout return
                if marker in new_stdout:
                    new_stdout = new_stdout.split(marker)[0]

    return {
        "completed": task.completed,
        "exit_code": task.exit_code,
        "stdout": new_stdout,
        "stderr": new_stderr,
        "cwd": cwd
    }

def ensure_cloudflared_binary():
    binary_name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    cloudflared_bin = os.path.join(SERVER_ROOT, binary_name)
    if not os.path.exists(cloudflared_bin):
        print("Downloading cloudflared...")
        download_url = (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            if os.name == "nt"
            else "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        )
        urllib.request.urlretrieve(download_url, cloudflared_bin)
        if os.name != "nt":
            os.chmod(cloudflared_bin, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return cloudflared_bin

def _safe_extract(zip_ref, dest_dir):
    """
    verify every archive member resolves to a path
    inside dest_dir before extracting anything. Raises ValueError if any
    entry would escape the destination directory (e.g. via '../' path
    segments, absolute paths, or symlink tricks in the member name).
    """
    dest_dir_real = os.path.realpath(dest_dir)
    for member in zip_ref.infolist():
        member_name = member.filename
        # Reject absolute paths and drive-letter tricks outright.
        if os.path.isabs(member_name) or (os.name == "nt" and ":" in member_name):
            raise ValueError(f"Unsafe path in archive: {member_name}")

        target_path = os.path.realpath(os.path.join(dest_dir_real, member_name))
        if not (target_path == dest_dir_real or target_path.startswith(dest_dir_real + os.sep)):
            raise ValueError(f"Unsafe path in archive: {member_name}")

    zip_ref.extractall(dest_dir)

@app.post("/upload/{project_name}")
async def upload_file(file: UploadFile = File(...), x_auth_token: str = Header(None), project_name: str = None):
    if project_name is None:
        raise HTTPException(status_code=400, detail="Project name is required")

    safe_project_name = os.path.basename(project_name.strip().replace("\\", "/"))
    if not safe_project_name or safe_project_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid project name")

    dest_dir = os.path.join(SERVER_ROOT, safe_project_name)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "repo.zip")

    try:
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            _safe_extract(zip_ref, dest_dir)

        os.remove(zip_path)
        return {"status": "success", "message": f"Extracted to {dest_dir}"}
    except ValueError as e:
        # Unsafe archive contents — reject, don't extract.
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return {"status": "error", "message": str(e)}

def start_cloudflare_tunnel():
    cloudflared_bin = ensure_cloudflared_binary()
    print("Starting Cloudflare Tunnel...")
    if CF_TUNNEL_TOKEN:
        tunnel_cmd = [cloudflared_bin, "tunnel", "run", "--token", CF_TUNNEL_TOKEN]
    else:
        tunnel_cmd = [cloudflared_bin, "tunnel", "--url", "http://localhost:8000"]

    tunnel_log = os.path.join(SERVER_ROOT, "tunnel.log")
    with open(tunnel_log, "w") as log_file:
        subprocess.Popen(tunnel_cmd, stdout=log_file, stderr=log_file)

    time.sleep(5)
    if not CF_TUNNEL_TOKEN and os.path.exists(tunnel_log):
        with open(tunnel_log, "r") as f:
            logs = f.read()
            for line in logs.split("\n"):
                if ".trycloudflare.com" in line:
                    url = [w for w in line.split() if ".trycloudflare.com" in w][0]
                    print(f"\nSERVER IS LIVE! Copy this URL for your local client:\n{url}\n")
                    break

def run_api_server():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="critical")
    server = uvicorn.Server(cfg)
    server.run()


if __name__ == "__main__":
    start_cloudflare_tunnel()
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    print(f"Uvicorn API server started in background thread. Root: {SERVER_ROOT}")
    try:
        while True:
            time.sleep(1)
    except Keyboard Interrupt:
        print("\nStopping API Server")
