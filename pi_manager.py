from Helpers import VideoSource as Source
from Helpers import inf_response as Res
from Helpers import Config, Offload_Controller
import multiprocessing, time, threading, requests, sys, os, psutil, csv
from pi_local_infer import infer_loop
import numpy as np
from ast import literal_eval

def request_offload(request, q, config, url='http://localhost:1234/infer'):
    t = time.time()
    try:
        req = requests.post(url, files = {\
            'image': request.image}, data = {'model': request.model}, timeout=config.get_latency_timeout())
        lat = time.time() - t
        data = literal_eval(req.content.decode())

        q.put(Res(request.id, data[:-1],False, lat))
    except Exception as E: #requests.exceptions.Timeout as T:
        lat = time.time() - t
        print('failed', E)
        config.add_timeout()
        q.put(Res(request.id, None, False, lat, False))


def capture_loop(q, num_to_test, shape, model = 'mobilenet'):
    images = Source(shape)
    for i in range(num_to_test):
        req = images.get_frame()
        req.model = model
        q.put(req)

def process_results(q, arr, num_to_test):
    for i in range(num_to_test):
        result = q.get()
        arr[result.id] = result

def change_network(config, done):
    for change in config.get_network_conditions():
        if done.is_set(): break
        #use -1 to disable
        rate = (change['rate'] if not change['rate'] == '-1' else '10000') + ' '
        loss = (change['loss'] if not change['loss'] == '-1' else '0') + ' '
        delay = (change['latency'] if not change['latency'] == '-1' else '0.1') + ' '
        jitter = (change['jitter'] if not change['jitter'] == '-1' else '0.1') + ' '
        config.set_current_net({'rate':rate ,\
            'loss': loss, 'latency': delay, 'jitter': jitter})
        os.system('sh update_net.sh ' + rate + loss + delay + jitter)
        time.sleep(float(change['wait_time']))


def measure_and_control(config, done, controller, stats_arr):
    d = 1 / config.get_measure_rate()
    st = time.time()
    cpu = None
    while not done.is_set():
        fps = config.measure_and_report_fps()
        print("FPS", fps)
        tps = config.measure_and_report_tps()
        print("TPS", tps)
        controller.control_and_update(tps)

        stats = {'time': time.time() - st,\
            'fps': fps, 'tps': tps, 'cpu': cpu, 'ops': config.get_offload_fps(),\
            'offload_fps': config.get_o_count()}
        net = config.get_current_net()
        if not net == None:
            stats.update(config.get_current_net())
            if None in stats.values():
                print('Stats not ready:', stats)
            else:
                stats_arr.append(stats)

        wait = d - ((time.time() - st) % d)
        #waits for wait and count CPU usage
        cpu = psutil.cpu_percent(wait)

def main(config_file):
    #create and parse config file
    config = Config(config_file)

    #create PID controller based on config specs
    controller = Offload_Controller(config)

    num_to_test = config.get_samples()
    frame_delay = 1 / config.get_source_fps()
    shape = config.get_shape()

    results_arr = np.ndarray((num_to_test,), dtype=Res)
    image_queue = multiprocessing.Queue(1)
    res_queue = multiprocessing.Queue()

    #an event that tells the measure and net threads to stop
    done_event = threading.Event()

    net_thread = threading.Thread(target=change_network,\
        args=(config, done_event))
    #image capture process
    cap_proc = multiprocessing.Process(target=capture_loop, \
        args=(image_queue, num_to_test, shape))

    #event that tells the local processing process to grab an image
    pull_from_queue_event = multiprocessing.Event()
    #event that tells this process that the local processing process
        #is ready for another frame
    infer_ready = multiprocessing.Event()
    #local inference process
    local_infer_proc = multiprocessing.Process( \
        target=infer_loop, args=(image_queue, res_queue,\
        infer_ready, pull_from_queue_event))
    local_infer_proc.start()

    #a thread that collects results
    res_thread = threading.Thread(target=process_results,\
        args=(res_queue, results_arr, num_to_test))


    #the measurement and controlling thread
        #runs in this process
    results_arr = []
    measure_thread = threading.Thread(target=measure_and_control, args=(config,\
        done_event, controller, results_arr))

    #warm up local processing
    pull_from_queue_event.set()
    infer_ready.wait()
    image_queue.put(Source(shape).get_frame())
    res_queue.get()
    infer_ready.wait()

    cap_proc.start()
    res_thread.start()
    print('STARTING')

    last_offload = time.time()

    offload_threads = []
    measure_thread.start()
    net_thread.start()
    start_time = time.time()

    #process images
    while config.get_procs() < num_to_test:
        t_since_last_offload = time.time() - last_offload
        need_to_wait = False
        #if next images should be offloaded
        if config.is_offloading_enabled()\
            and t_since_last_offload - (1/config.get_offload_fps()) > -0.003:

            req = image_queue.get()
            last_offload = time.time()
            config.add_proc()
            config.add_o_count()
            thread = threading.Thread(target=request_offload, args=(req, res_queue, config))
            thread.start()
            offload_threads.append(thread)
            need_to_wait = True

        #If the next frame can be processed locally
        elif infer_ready.is_set() and \
            ((1/config.get_offload_fps()) > frame_delay\
            or not config.is_offloading_enabled()):

            config.add_proc()
            #tell local processing process to process an image
            pull_from_queue_event.set()
            need_to_wait = True
        
        #wait until the next frame & continue the loop
        if need_to_wait:
            wait = frame_delay - ((time.time() - start_time) % frame_delay)
            time.sleep(wait)
    print("DONE")
    #wait until all results are in
    res_thread.join()
    
    #all offload threads should be done by this time
    for t in offload_threads:
        t.join()
    total_time = time.time() - start_time

    #stop measuring
    done_event.set()
    measure_thread.join()
    net_thread.join()
    os.system('sh reset_net.sh')
    #this FPS is not entirely accurate, the measured fps is more accurate
    print("Total time:", total_time, "FPS =", num_to_test / total_time)
    print("Offload %:", config.get_o_count() / num_to_test)
    #print(results_arr)

    with open('metrics/' + str(time.time()) + '.csv', 'w', newline='') as csvfile:
        fieldnames = results_arr[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for r in results_arr:
            writer.writerow(r)

    #kill local processing 
    local_infer_proc.kill()
    local_infer_proc.join()
    local_infer_proc.close()
    
    cap_proc.join()



if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
    except:
        config_file = 'default_config.ini'
    print(config_file)
    main(config_file)
