"""Pub/Sub publisher + subscriber for async document-ingestion jobs.

Real mode publishes JSON messages to the configured topic and (optionally)
runs a background streaming-pull subscriber. On Cloud Run, prefer the push
endpoint (POST /pubsub/push in app.main) over pull.

Mock mode (VERTEX_AI_MOCK=true) swaps Pub/Sub for an in-memory queue drained
by a daemon worker thread, so async ingestion behaves the same locally.
"""

import json
import logging
import queue
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level so every PubSubHandler instance shares one queue/worker,
# mirroring how a real topic is shared by all publishers and subscribers.
_mock_queue: "queue.Queue[dict]" = queue.Queue()
_mock_worker_started = False
_mock_callback = None
_mock_lock = threading.Lock()


def reset_mock_queue() -> None:
    """Drain pending mock messages (used by tests for isolation)."""
    while True:
        try:
            _mock_queue.get_nowait()
            _mock_queue.task_done()
        except queue.Empty:
            return


def _mock_worker() -> None:
    while True:
        message = _mock_queue.get()
        callback = _mock_callback
        try:
            if callback is not None:
                callback(message["job_id"])
        except Exception:
            logger.exception("Mock subscriber failed on job %s", message.get("job_id"))
        finally:
            _mock_queue.task_done()


class PubSubHandler:
    def __init__(self):
        settings = get_settings()
        self.mock = settings.vertex_ai_mock
        self.project_id = settings.gcp_project_id
        self.topic_name = settings.pubsub_topic
        self.subscription_name = settings.pubsub_subscription
        self._publisher = None

    # ------------------------------------------------------------------ #
    # Publishing                                                          #
    # ------------------------------------------------------------------ #

    def publish_ingestion_job(self, job_id: str) -> str:
        """Publish an ingestion job message; returns the Pub/Sub message id."""
        if self.mock:
            _mock_queue.put({"job_id": job_id})
            return f"mock-{job_id}"
        publisher = self._get_publisher()
        topic_path = publisher.topic_path(self.project_id, self.topic_name)
        data = json.dumps({"job_id": job_id}).encode("utf-8")
        future = publisher.publish(topic_path, data)
        message_id = future.result(timeout=30)
        logger.info("Published job %s as message %s", job_id, message_id)
        return message_id

    def _get_publisher(self):
        if self._publisher is None:
            from google.cloud import pubsub_v1

            self._publisher = pubsub_v1.PublisherClient()
        return self._publisher

    # ------------------------------------------------------------------ #
    # Subscribing                                                         #
    # ------------------------------------------------------------------ #

    def start_subscriber(self, callback) -> None:
        """Start consuming ingestion jobs in the background.

        ``callback`` receives the job_id of each message. Mock mode uses the
        shared in-memory queue; real mode opens a streaming pull. Calling this
        again just swaps in the new callback.
        """
        global _mock_worker_started, _mock_callback
        if self.mock:
            with _mock_lock:
                _mock_callback = callback
                if not _mock_worker_started:
                    threading.Thread(
                        target=_mock_worker, name="mock-pubsub-worker", daemon=True
                    ).start()
                    _mock_worker_started = True
            return

        from google.cloud import pubsub_v1

        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(
            self.project_id, self.subscription_name
        )

        def _on_message(message):
            try:
                payload = json.loads(message.data.decode("utf-8"))
                callback(payload["job_id"])
                message.ack()
            except Exception:
                logger.exception("Failed to process message %s", message.message_id)
                message.nack()

        subscriber.subscribe(subscription_path, callback=_on_message)
        logger.info("Streaming-pull subscriber started on %s", subscription_path)

    @staticmethod
    def parse_push_message(envelope: dict) -> str:
        """Extract job_id from a Pub/Sub push delivery envelope."""
        import base64

        data = envelope["message"]["data"]
        payload = json.loads(base64.b64decode(data).decode("utf-8"))
        return payload["job_id"]
