from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
from VideoSource import inf_response as Res
import multiprocessing, time, threading
from pi_local_infer import infer_loop


def capture_loop(q, num_to_test, shape, frame_delay, model = 'mobilenet'):
    images = Source(shape)
    for i in range(num_to_test):
        req = images.get_frame()
        req.model = model
        q.put(req)

def change_offload(o):
    time.sleep(10)
    o['frame_delay'] = 1 / 30



def main():
    num_to_test = 250
    off = {'frame_delay': 1/3}
    fps = 30
    frame_delay = 1 / fps
    shape = (224,224)
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
    c = threading.Thread(target=change_offload, args=(off,))

    #warm up
    infer_ready.wait()
    req_queue.put(Source(shape).get_frame())
    res_queue.get()
    print('starting')

    last_offload = time.time()


    o_count = 0
    processed = 0
    c.start()
    start_time = time.time()
    print('start time', start_time)
    while processed < num_to_test:
        t_since_last_offload = time.time() - last_offload
        if t_since_last_offload - off['frame_delay'] > -0.003:
            #print(last)
            req = image_queue.get()
            #print(last_offload)
            last_offload = time.time()
            processed += 1
            o_count += 1
            res_queue.put(Res(req.id, "offloaded", False, time.time()))
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

    for i in range(num_to_test):
        try:
            results = res_queue.get()
            print(i, results.classes, 'times', results.timestamp)
        except:
            break
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
