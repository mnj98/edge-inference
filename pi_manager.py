from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
from VideoSource import inf_response as Res
import multiprocessing, time
from pi_local_infer import infer_loop


def capture_loop(q, num_to_test, shape, frame_delay, model = 'mobilenet'):
    images = Source(shape)
    start_time = time.time()
    for i in range(num_to_test):
        req = images.get_frame()
        req.model = model
        #q.put(req)
        wait = frame_delay - ((time.time() - start_time) % frame_delay)
        #print(wait)
        time.sleep(wait)
        q.put(req)


def main():
    num_to_test = 250
    fps = 30
    shape = (224,224)
    #image_queue = multiprocessing.Queue(1)
    req_queue = multiprocessing.Queue(1)
    res_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(req_queue, num_to_test, shape, 1/fps))
    infer_ready = multiprocessing.Event()
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(req_queue, res_queue, infer_ready))
    local_infer_proc.start()
    cap_proc.start()


    #warm up TODO: reset the id on the image source side so
        #ids line up for accuracy measurement
    infer_ready.wait()
    req_queue.put(Source(shape).get_frame())
    res_queue.get()
    print('starting')
    off_frame_delay = 1/ 4
    start_time = time.time()
    #for i in range(int(num_to_test)):
    o_count = 0
    while True:
        try:
            wait = off_frame_delay - ((time.time() - start_time) % off_frame_delay)
            r = req_queue.get(timeout=wait)
            wait = off_frame_delay - ((time.time() - start_time) % off_frame_delay)
            res_queue.put(Res(r.id, 'Offloaded', False))
            o_count+=1
            time.sleep(wait)
        except:
            if res_queue.qsize() >= num_to_test: break
            continue
    
    #for i in range(num_to_test):
        #image = image_queue.get()
        #req = Req(i, image)

        #infer_ready.wait()
        
        #req_queue.put(req)
        #print(req_queue.qsize())
    #print('done with loop')
    for i in range(num_to_test):
        try:
            print(i, res_queue.get().classes)
        except:
            break
    total_time = time.time() - start_time
    print("Total time:", total_time, "FPS =", num_to_test / total_time)
    print("Offload %:", o_count / num_to_test) 
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    cap_proc.join()
    


if __name__ == "__main__":
    main()
