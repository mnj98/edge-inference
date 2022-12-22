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
            'image': request.image}, data = {'model': request.model}, timeout=5)
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

def change_offload(o):
    time.sleep(10)
    o['frame_delay'] = 1 / 60



def main():
    num_to_test = 500
    off = {'frame_delay': 1/60}
    fps = 70
    frame_delay = 1 / fps
    shape = (224,224)
    results_arr = np.ndarray((num_to_test,), dtype=Res)
    image_queue = multiprocessing.Queue(1)
    req_queue = multiprocessing.Queue(1)
    res_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(image_queue, num_to_test, shape, 1/fps))
    infer_ready = multiprocessing.Event()
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(req_queue, res_queue, infer_ready))
    local_infer_proc.start()
    cap_proc.start()
    res_thread = threading.Thread(target=process_results, args=(res_queue, results_arr, num_to_test))
    


    c = threading.Thread(target=change_offload, args=(off,))

    #warm up
    infer_ready.wait()
    req_queue.put(Source(shape).get_frame())
    res_queue.get()

    res_thread.start()
    print('starting')

    last_offload = time.time()


    o_count = 0
    processed = 0
    offload_threads = []
    c.start()
    start_time = time.time()
    print('start time', start_time)

    while processed < num_to_test:
        t_since_last_offload = time.time() - last_offload
        if t_since_last_offload - off['frame_delay'] > -0.003: #offload
            #print(last)
            req = image_queue.get()
            #print(last_offload)
            last_offload = time.time()
            processed += 1
            o_count += 1
            thread = threading.Thread(target=request_offload, args=(req, res_queue, 0.5))
            thread.start()
            offload_threads.append(thread)
            #res_queue.put(Res(req.id, "offloaded", False, time.time()))
            wait = frame_delay - ((time.time() - start_time) % frame_delay)
            time.sleep(wait)
            continue
        if infer_ready.is_set() and off['frame_delay'] != frame_delay:
            req_queue.put(image_queue.get())
            processed += 1
        #print(t_since_last_offload)
        wait = frame_delay - ((time.time() - start_time) % frame_delay)
        time.sleep(wait)
        continue

    res_thread.join()
    for t in offload_threads:
        t.join()
    total_time = time.time() - start_time
    print("Total time:", total_time, "FPS =", num_to_test / total_time)
    print("Offload %:", o_count / num_to_test) 
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    cap_proc.join()
    c.join()



if __name__ == "__main__":
    main()
