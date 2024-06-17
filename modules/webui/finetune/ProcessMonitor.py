import os
import subprocess
import sys
import threading


class ProcessMonitor:
    def __init__(self):
        self.process = None
        self.stdout = ""
        self.stderr = ""
        self.lock = threading.Lock()

    def start_process(self, command):
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        # Set pipes to non-blocking mode
        fd_out = self.process.stdout.fileno()
        fd_err = self.process.stderr.fileno()

        if sys.platform != "win32":
            import fcntl

            fl_out = fcntl.fcntl(fd_out, fcntl.F_GETFL)
            fl_err = fcntl.fcntl(fd_err, fcntl.F_GETFL)
            fcntl.fcntl(fd_out, fcntl.F_SETFL, fl_out | os.O_NONBLOCK)
            fcntl.fcntl(fd_err, fcntl.F_SETFL, fl_err | os.O_NONBLOCK)

        # Start threads to read stdout and stderr
        threading.Thread(target=self._read_stdout).start()
        threading.Thread(target=self._read_stderr).start()

    def _read_stdout(self):
        while self.process is not None and self.process.poll() is None:
            try:
                output = self.process.stdout.read()
                if output:
                    with self.lock:
                        self.stdout += output
            except:
                pass

    def _read_stderr(self):
        while self.process is not None and self.process.poll() is None:
            try:
                error = self.process.stderr.read()
                if error:
                    with self.lock:
                        self.stderr += error
            except:
                pass

    def get_output(self):
        with self.lock:
            return self.stdout, self.stderr

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process = None


if __name__ == "__main__":
    import time

    pm = ProcessMonitor()
    pm.start_process(
        [
            "python",
            "-u",
            "-c",
            "import time; [print(i) or time.sleep(1) for i in range(5)]",
        ]
    )

    while pm.process and pm.process.poll() is None:
        stdout, stderr = pm.get_output()
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)
        time.sleep(1)

    stdout, stderr = pm.get_output()
    print("Final STDOUT:", stdout)
    print("Final STDERR:", stderr)
