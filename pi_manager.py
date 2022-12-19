from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
import multiprocessing
from pi_local_infer import infer_loop


def capture_loop(q, num_to_test, shape):
    images = Source(shape)

    for i in range(num_to_test):
        q.put(images.get_frame())


def main():
    num_to_test = 5000
    shape = (223,224)
    image_queue = multiprocessing.Queue(1)
    req_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(image_queue, num_to_test, shape))
    infer_ready = multiprocessing.Event()
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(req_queue, res_queue, infer_ready))
    local_infer_proc.start()
    
    cap_proc.start()
    for i in range(num_to_test):
        image = image_queue.get()
        req = Req(i, image)

        infer_ready.wait()
        req_queue.put(req)

    for i in range(num_to_test):
        print(res_queue.get().preds)
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    cap_proc.join()
    


if __name__ == "__main__":
    main()