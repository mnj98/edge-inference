from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
from VideoSource import inf_response as Res
import multiprocessing, time, threading, requests
from pi_local_infer import infer_loop
import numpy as np
from ast import literal_eval

def request_offload(request, q, timeout, url='http://localhost:1234/infer'):
    try:
        req = requests.post(url, files = {\
            'image': request.image}, data = {'model': request.model}, timeout=1)
        data = literal_eval(req.content.decode())

        q.put(Res(request.id, data[:-1],False, time.time()))
    except Exception as E: #requests.exceptions.Timeout as T:
        print('failed', E)
        q.put(Res(request.id, None, False, time.time(), False))


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

def measure(config, lock, done):
    d = 1 / config['measure_rate']
    st = time.time()
    while not done.is_set():
        with lock:
            config['processed_in_time'].append((time.time(), config['processed']))
            if len(config['processed_in_time']) > 1:
                prev = config['processed_in_time'][-2]
                curr = config['processed_in_time'][-1]
                print("FPS:", (curr[1] - prev[1]) / (curr[0] - prev[0]))
            if len(config['processed_in_time']) > 15:
                config['enabled'] = False
        wait = d - ((time.time() - st) % d)
        time.sleep(wait)
def get_processed(config, lock):
    with lock:
        return config['processed']

def main():
    num_to_test = 500
    config_lock = threading.Lock()
    offload_config = {'frame_delay': 1/10,\
         'enabled': True,\
         'processed': 0,\
         'processed_in_time': [],\
         'measure_rate': 3}
    fps = 30
    frame_delay = 1 / fps
    shape = (224,224)
    results_arr = np.ndarray((num_to_test,), dtype=Res)
    image_queue = multiprocessing.Queue(1)
    req_queue = multiprocessing.Queue(1)
    res_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(image_queue, num_to_test, shape, 1/fps))
    pull_from_queue_event = multiprocessing.Event()
    infer_ready = multiprocessing.Event()
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(image_queue, res_queue, infer_ready, pull_from_queue_event))
    local_infer_proc.start()
    res_thread = threading.Thread(target=process_results, args=(res_queue, results_arr, num_to_test))
    

    measure_done_event = threading.Event()
    measure_thread = threading.Thread(target=measure, args=(offload_config,config_lock,\
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

    while get_processed(offload_config, config_lock) < num_to_test:
        t_since_last_offload = time.time() - last_offload
        if offload_config['enabled']\
            and t_since_last_offload - offload_config['frame_delay'] > -0.003: #offload
            req = image_queue.get()
            last_offload = time.time()
            with config_lock:
                offload_config['processed'] += 1
            o_count += 1
            thread = threading.Thread(target=request_offload, args=(req, res_queue, 0.5))
            thread.start()
            offload_threads.append(thread)

        elif infer_ready.is_set() and \
            (offload_config['frame_delay'] > frame_delay or not offload_config['enabled']):
            with config_lock:
                offload_config['processed'] += 1
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
