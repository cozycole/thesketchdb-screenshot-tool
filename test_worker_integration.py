import tempfile
import time
import os

import docker
from docker import errors
from docker import types as docker_types
from celery import Celery

client = docker.from_env()

def test_worker_integration(tmp_path):
    app, worker_container, redis_container, task = None, None, None, None
    try:
        # print("Building screenshot worker container")
        # _, _ = client.images.build(path=".", tag="screenshot_worker:test")
        print("Creating Docker network")
        try:
            client.networks.create("test-network", driver="bridge")
        except errors.APIError:
            # Network might already exist
            client.networks.get("test-network")

        # 2. Start redis + shared volume
        volume_path = os.path.join(tmp_path, "shared")
        os.makedirs(volume_path, exist_ok=True)

        print("Spinning up redis container")
        redis_container = client.containers.run(
            "redis:7",
            name="redis-test",
            detach=True,
            network="test-network",
            ports={"6379/tcp": 6379},
        )

        print("Spinning up worker container")
        worker_container = client.containers.run(
            "screenshot-gpu",
            name="worker-test",
            detach=True,
            network="test-network",
            environment={"CELERY_BROKER_URL": "redis://redis-test:6379/0", "MODEL_PATH" : "./model.pth"},
            volumes={str(volume_path): {"bind": "/shared", "mode": "rw"}},
            device_requests=[
                docker_types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        )
        breakpoint()

        print("Connecting celery to redis")
        app = Celery(
            "test",
            broker="redis://localhost:6379/0",
            backend="redis://localhost:6379/0",
        )

        print("Queueing task")
        task = app.send_task(
            "screenshot",
            args=["https://thesketchdb-testing.nyc3.cdn.digitaloceanspaces.com/outrageous-test.mp4", "/shared/output"],
        )

        start = time.time()
        while task.status not in ("SUCCESS", "FAILURE"):
            if time.time() - start > 120:
                raise TimeoutError("Task did not finish in time")
            time.sleep(2)

        if task.status != "SUCCESS":
            print(task.status)
            breakpoint()

        print("Task ran successfully! Output:", task.result)
        breakpoint()

    finally:
        try:
            if task:
                task.forget()
                del task
            if app:
                print("closing app")
                app.close()
        except Exception as e:
            print(f"Error closing Celery app: {e}")


        time.sleep(1)

        if worker_container:
            print("removing worker container")
            worker_container.stop()
            worker_container.remove()
        if redis_container:
            print("removing redis")
            redis_container.stop()
            redis_container.remove()

if __name__ == "__main__":
    tmp_dir = tempfile.mkdtemp(prefix="screenshot_output_")
    print(f"Temp dir path: {tmp_dir}")
    test_worker_integration(tmp_dir)
