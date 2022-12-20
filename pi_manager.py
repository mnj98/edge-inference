from VideoSource import VideoSource as Source
from VideoSource import inf_request as Req
import multiprocessing, time
from pi_local_infer import infer_loop


def capture_loop(q, num_to_test, shape):
    images = Source(shape)

    for i in range(num_to_test + 1):
        q.put(images.get_frame())


def main():
    num_to_test = 50
    shape = (224,224)
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


    #warm up TODO: reset the id on the image source side so
        #ids line up for accuracy measurement
    infer_ready.wait()
    req_queue.put(Req(0, image_queue.get()))
    res_queue.get()


    start_time = time.time()
    for i in range(num_to_test):
        image = image_queue.get()
        req = Req(i, image)

        infer_ready.wait()
        req_queue.put(req)
    print('done with loop')
    for i in range(num_to_test):
        print(res_queue.get().classes)
    total_time = time.time() - start_time
    print("Total time:", total_time, "FPS =", num_to_test / total_time)
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    cap_proc.join()
    


if __name__ == "__main__":
    main()
