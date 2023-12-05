import multiprocessing
import struct
import time
from multiprocessing import shared_memory

import numpy as np
import onnxruntime as ort
import zmq

from data_processing import preprocess_sample
from hyperparameters import DATA_LEN
from worker import worker


def wrist_angle_worker(shared_name, lock):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect("tcp://localhost:5556")

    existing_shm = shared_memory.SharedMemory(name=shared_name)
    np_array = np.ndarray((DATA_LEN, 3), dtype=np.float64, buffer=existing_shm.buf)

    while True:
        # Send string 'wrist_angle' to server
        socket.send(b'wrist_angle')
        # Receive the response from the server
        response = socket.recv()
        # Check if response is empty
        if len(response) == 0:
            time.sleep(0.01)
            continue

        float_array = struct.unpack('{}f'.format(len(response) // 4), response)
        float_array = np.array(float_array, dtype=np.float64).reshape((DATA_LEN, 3))

        with lock:
            np_array[:] = float_array[:]

        time.sleep(0.01)

    socket.close()
    context.term()
    existing_shm.close()

def prep_worker(prep_q, shared_name, lock):
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect("tcp://localhost:5555")

    is_prep_running = True

    # Load ONNX model
    ort_sess = ort.InferenceSession('model_rnn.onnx')
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    np_array = np.ndarray((DATA_LEN, 3), dtype=np.float64, buffer=existing_shm.buf)

    while is_prep_running:
        while not (prep_q.empty()):
            emgs = prep_q.get()
            processed = preprocess_sample(np.array(emgs))
            # Preprocess the data
            # with lock:
            #     processed = preprocess_sample(np.array(emgs), np_array)
                # print("Array: " + str(np_array[-1]))

            processed = np.expand_dims(processed, axis=0)
            outputs = ort_sess.run(None, {'input': processed})
            predicted = outputs[0][0]

            # Convert the data to a flat list of floats - for inference
            flat_data = [item for sublist in predicted for item in sublist]
            # Optionally print the data
            # print("Sending result data:", flat_data)
            # Pack the floats as binary data
            packed_data = struct.pack('{}f'.format(len(flat_data)), *flat_data)
            # Send via zmq
            socket.send(packed_data)
            # Receive the response from the server
            response = socket.recv()
            # print("Received response:", response.decode())

    # Close the socket and context
    socket.close()
    context.term()
    existing_shm.close()


# ------------ Myo Setup ---------------
q = multiprocessing.Queue()
prep_q = multiprocessing.Queue()

if __name__ == "__main__":
    shm = shared_memory.SharedMemory(create=True, size=DATA_LEN * 3 * np.float64().itemsize)
    # Create a lock to synchronize array access
    lock = multiprocessing.Lock()

    # Wrap shared memory block as a NumPy array
    np_array = np.ndarray((DATA_LEN, 3), dtype=np.float64, buffer=shm.buf)

    # wrist_angle_p = multiprocessing.Process(target=wrist_angle_worker, args=(shm.name,lock,))
    # wrist_angle_p.start()

    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    prep_p = multiprocessing.Process(target=prep_worker, args=(prep_q, shm.name, lock,))
    prep_p.start()

    # emgs is a list of lists acting like a queue
    emgs = []

    is_connected = False

    avg_fps = 0
    counter = 0

    timer = time.time()

    try:
        while True:
            while not (q.empty()):
                counter += 1

                emg = list(q.get())
                emgs.append(emg)
                if len(emgs) < DATA_LEN:
                    continue

                prep_q.put(emgs.copy())

                # Remove the oldest sample
                del emgs[0]

    except KeyboardInterrupt:
        print("Avg fps: ", counter / (time.time() - timer))
        print("Quitting")
        is_prep_running = False
        time.sleep(1)
        quit()
