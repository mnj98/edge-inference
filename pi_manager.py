from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
from VideoSource import inf_response as Res
from VideoSource import Config
import multiprocessing, time, threading, requests, sys
from pi_local_infer import infer_loop
import numpy as np
from ast import literal_eval

def request_offload(request, q, config, timeout, url='http://localhost:1234/infer'):
    t = time.time()
    try:
        req = requests.post(url, files = {\
            'image': request.image}, data = {'model': request.model}, timeout=timeout)
        lat = time.time() - t
        data = literal_eval(req.content.decode())

        q.put(Res(request.id, data[:-1],False, lat))
    except Exception as E: #requests.exceptions.Timeout as T:
        lat = time.time() - t
        print('failed', E)
        config.add_timeout()
        q.put(Res(request.id, None, False, lat, False))


def capture_loop(q, num_to_test, shape, frame_delay, model = 'mobilenet'):
    images = Source(shape)
    for i in range(num_to_test):
        req = images.get_frame()
        req.model = model
        q.put(req)

def process_results(q, arr, num_to_test):
    for i in range(num_to_test):
        result = q.get()
        arr[result.id] = result
        #print('lat for', result.id, 'is', result.latency)

def measure(config, done):
    d = 1 / config.get_measure_rate()
    st = time.time()
    while not done.is_set():
        print("FPS", config.measure_and_report_fps())
        print("TPS", config.measure_and_report_tps())
        wait = d - ((time.time() - st) % d)
        time.sleep(wait)

def main():
    num_to_test = int(sys.argv[1])
    timeout = float(sys.argv[2])
    #config_lock = threading.Lock()
    #offload_config = {'frame_delay': 1/31,\
    #     'enabled': True,\
    #     'processed': 0,\
    #     'processed_in_time': [],\
    #     'measure_rate': 3}
    #fps = 30
    config = Config(source_fps=60, offload_fps=60, mps=3)
    #config.disable_offloading()
    frame_delay = 1 / config.sfps
    shape = config.shape
    results_arr = np.ndarray((num_to_test,), dtype=Res)
    image_queue = multiprocessing.Queue(1)
    req_queue = multiprocessing.Queue(1)
    res_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(image_queue, num_to_test, shape, 1/config.sfps))
    pull_from_queue_event = multiprocessing.Event()
    infer_ready = multiprocessing.Event()
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(image_queue, res_queue, infer_ready, pull_from_queue_event))
    local_infer_proc.start()
    res_thread = threading.Thread(target=process_results, args=(res_queue, results_arr, num_to_test))
    

    measure_done_event = threading.Event()
    measure_thread = threading.Thread(target=measure, args=(config,\
        measure_done_event))
    pull_from_queue_event.set()
    #warm up
    infer_ready.wait()
    image_queue.put(Source(shape).get_frame())
    res_queue.get()

    
    time.sleep(1)
    cap_proc.start()
    res_thread.start()
    print('starting')

    last_offload = time.time()


    o_count = 0
    #processed = 0
    offload_threads = []
    measure_thread.start()
    start_time = time.time()
    print('start time', start_time)

    while config.get_procs() < num_to_test:
        t_since_last_offload = time.time() - last_offload
        if config.is_offloading_enabled()\
            and t_since_last_offload - (1/config.get_offload_fps()) > -0.003: #offload
            req = image_queue.get()
            last_offload = time.time()
            config.add_proc()
            o_count += 1
            thread = threading.Thread(target=request_offload, args=(req, res_queue, config, timeout))
            thread.start()
            offload_threads.append(thread)

        elif infer_ready.is_set() and \
            ((1/config.get_offload_fps()) > frame_delay\
            or not config.is_offloading_enabled()):
            config.add_proc()
            pull_from_queue_event.set()
        wait = frame_delay - ((time.time() - start_time) % frame_delay)
        time.sleep(wait)
        #with config_lock:
        #    print(offload_config['processed_in_time'])
        continue

    res_thread.join()
    
    for t in offload_threads:
        t.join()
    total_time = time.time() - start_time
    measure_done_event.set()
    measure_thread.join()
    print("Total time:", total_time, "FPS =", num_to_test / total_time)
    print("Offload %:", o_count / num_to_test)
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    cap_proc.join()



if __name__ == "__main__":
    main()
