import os
import signal
import subprocess
import time
from rerun.experimental import ViewerClient
import rerun.blueprint as rrb
import rerun as rr


class Screenshot:
    """Captures screenshots from a Rerun viewer running in a separate process.

    Launches a Rerun viewer as a subprocess and connects to it via gRPC.
    Automatically manages the viewer process lifecycle, ensuring clean shutdown
    of the viewer window.

    Args:
        window_size: Window dimensions as "WIDTHxHEIGHT" (default: "600x600")
        port: gRPC server port (default: 9777)
    """

    def __init__(self, window_size: str = "600x600", port: int = 9777) -> None:
        """Initialize the screenshot capturer with a subprocess-based Rerun viewer.

        Args:
            window_size: Window resolution as "WIDTHxHEIGHT", e.g. "600x600"
            port: gRPC port to use for communication (default: 9777)
        """
        self.port = port
        self.process = None

        # Launch Rerun viewer as subprocess
        # start_new_session=True creates a new process group, so we can kill
        # the entire tree (including the detached viewer window) later.
        cmd = [
            "rerun",
            "--window-size",
            window_size,
            "--port",
            str(port),
            "--expect-data-soon",
        ]
        self.process = subprocess.Popen(cmd, start_new_session=True)

        # Register cleanup to run at exit to ensure window closes
        # atexit.register(self.stop)

        # Give the server time to start
        time.sleep(2)

        # Connect to the running viewer via gRPC
        rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")

        self.viewer = ViewerClient(addr=f"127.0.0.1:{port}")
        self.view = rrb.Spatial3DView(
            name="USD Render",
            # background=[89, 32, 185],
            eye_controls=rrb.EyeControls3D(
                position=(-1.32, -0.72, 0.72), look_target=(0, 0, -0.25)
            ),
        )
        rr.send_blueprint(rrb.Blueprint(self.view, collapse_panels=True))

    def take(self, path: str) -> None:
        """Save a screenshot of the current view.

        Args:
            path: File path where the screenshot will be saved
        """
        time.sleep(1)  # Ensure the viewer has rendered the latest data
        print(f"Saving screenshot to {path} with view ID {self.view.id}")
        self.viewer.save_screenshot(path, view_id=self.view.id)
        # Wait until the file is actually written to disk
        while not os.path.exists(path):
            time.sleep(0.1)

    def stop(self) -> None:
        """Stop the Rerun viewer process and all its children."""
        # Unregister to avoid double calls
        # try:
        #     atexit.unregister(self.stop)
        # except Exception:
        #     pass

        if self.process:
            # Kill the process group to ensure grandchildren (the actual viewer window) are killed
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

            self.process = None

    def __del__(self) -> None:
        """Ensure the subprocess is cleaned up when the object is destroyed."""
        self.stop()
