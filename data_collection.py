import multiprocessing
import struct
import time

import numpy as np
import zmq

from worker import worker

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

# True if just recording locally, False if sending to server (Unity)
IS_LOCAL_RECORDING = False

if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect("tcp://localhost:5555")

    # emgs is a list of lists acting like a queue
    all_emgs = []
    is_connected = False
    timer = time.time()
    avg_fps = 0
    counter = 0
    try:
        while True:
            while not (q.empty()):
                counter += 1
                emg = list(q.get())

                # Convert the data to a flat list
                flat_data = [float(item) for item in emg]

                # Pack the floats as binary data
                packed_data = struct.pack('{}f'.format(len(flat_data)), *flat_data)

                # Send via zmq
                socket.send(packed_data)

                # Receive the response from the server
                response = socket.recv()

                if IS_LOCAL_RECORDING:
                    all_emgs.append(flat_data.copy())

                # if time.time() - timer > 20:
                #     raise KeyboardInterrupt

                now = time.time()

    except KeyboardInterrupt:
        print("Avg fps: ", counter / (time.time() - timer))
        print("Quitting")
        # Close the socket and context
        socket.close()
        context.term()

        if IS_LOCAL_RECORDING:
            # Save all_emgs to a csv file
            np.savetxt("data/emgs_local.csv", np.array(all_emgs, dtype=float), delimiter=",")
        exit(0)
