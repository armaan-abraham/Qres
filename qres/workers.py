import torch.multiprocessing as mp
import logging
from qres.structure_prediction import StructurePredictor


def structure_predictor_worker(task_queue: mp.Queue, result_queue: mp.Queue, device_id: int):
    # Set up logging for the worker
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"Worker-{device_id}")
    logger.info(f"Starting worker on device cuda:{device_id}")

    # Set device
    device = f'cuda:{device_id}'

    # Instantiate the structure predictor on the assigned GPU
    predictor = StructurePredictor(device=device)

    while True:
        # Get task
        task = task_queue.get()
        if task is None:
            # Shutdown signal
            logger.info("Shutting down.")
            break

        # Unpack the task
        task_id, sequences = task

        # Perform structure prediction
        try:
            pdbs = predictor.predict_structure(sequences)
            confidences = [
                predictor.overall_confidence_from_pdb(pdb)
                for pdb in pdbs
            ]
        except Exception as e:
            logger.exception("Error during prediction.")
            confidences = [0.0] * len(sequences)  # Handle errors gracefully

        # Put the result in the result queue
        result_queue.put((task_id, confidences))


class StructurePredictorPool:
    def __init__(self, num_workers: int, device_ids: List[int]):
        assert num_workers == len(device_ids), "Number of workers must match number of device IDs"
        self.num_workers = num_workers
        self.tasks = mp.Queue()
        self.results = mp.Queue()

        # Start worker processes
        self.workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=structure_predictor_worker,
                args=(self.tasks, self.results, device_ids[i])
            )
            p.start()
            self.workers.append(p)

        self.task_counter = 0

    def predict_structure(self, sequences: List[str]) -> List[float]:
        num_sequences = len(sequences)
        num_workers = min(self.num_workers, num_sequences)
        sequences_per_worker = num_sequences // num_workers
        remaining = num_sequences % num_workers

        task_ids = []
        start = 0
        for i in range(num_workers):
            end = start + sequences_per_worker + (1 if i < remaining else 0)
            seq_chunk = sequences[start:end]
            task_id = self.task_counter
            self.task_counter += 1
            self.tasks.put((task_id, seq_chunk))
            task_ids.append((task_id, len(seq_chunk)))
            start = end

        # Collect results
        confidences = []
        results_received = 0
        while results_received < len(task_ids):
            task_id, confs = self.results.get()
            confidences.extend(confs)
            results_received += 1

        return confidences

    def shutdown(self):
        # Send shutdown signal to workers
        for _ in self.workers:
            self.tasks.put(None)
        # Join worker processes
        for p in self.workers:
            p.join()
