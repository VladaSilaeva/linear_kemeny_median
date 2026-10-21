import time
import multiprocessing as mp

def _run_with_queue(func, args, kwargs, queue):
    """Выполняет функцию и кладёт в очередь (статус, время, результат) или (статус, ошибка)"""
    try:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        queue.put(('time', elapsed, result))
    except Exception as e:
        queue.put(('error', e, None))

def measure_time_with_result(func, timeout, args=(), kwargs=None):
    """Возвращает (время, результат) или (None, None) при таймауте/ошибке"""
    if kwargs is None:
        kwargs = {}
    queue = mp.Queue()
    proc = mp.Process(target=_run_with_queue, args=(func, args, kwargs, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, None  # таймаут

    try:
        status, value, result = queue.get_nowait()
        if status == 'time':
            return value, result
        else:
            # Ошибка внутри функции (можно залогировать, если нужно)
            return None, value
    except mp.queues.Empty:
        return None, None

def my_function(n, m, k):
    # Пример: медленный расчёт
    total = 0
    for i in range(n * m * k):
        total += i ** 0.5
    return total