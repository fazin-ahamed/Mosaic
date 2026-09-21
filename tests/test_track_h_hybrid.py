"""CPU-only regression for H1 task transport and executed-work accounting."""
from queue import Queue
from threading import Event, Thread
from unittest.mock import Mock, patch

from mosaicvm.bench import track_h_hybrid as h1


def test_h1_gu_accounting_and_separate_request_result_queues():
    assert h1.cpu_gu_task_flops() == 2 * 3 * 4096 * 4096
    request_q, result_q = Queue(), Queue()
    ready, stop = Event(), Event()
    kernel = Mock()
    library = Mock(vnni_gemm_tiled=kernel)
    # Tiny dimensions and a fake C kernel exercise the actual worker loop
    # without loading VNNI instructions, allocating large weights, or using CUDA.
    with patch.object(h1, 'HIDDEN', 4), patch.object(h1, 'INTER', 4), \
            patch.object(h1.ctypes, 'CDLL', return_value=library):
        worker = Thread(target=h1.cpu_vnni_worker,
                        args=(request_q, result_q, ready, stop), daemon=True)
        worker.start()
        try:
            assert ready.wait(5)
            for tasks in (2, 3):
                request_q.put(tasks)
                tag, elapsed, completed = result_q.get(timeout=5)
                assert (tag, completed) == ('cpu_done', tasks)
                assert elapsed >= 0
            assert kernel.call_count == 5
            assert request_q.empty() and result_q.empty()
        finally:
            request_q.put(None)
            worker.join(5)
            stop.set()
        assert not worker.is_alive()
