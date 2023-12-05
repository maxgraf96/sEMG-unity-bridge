# Multimodal Hand Tracking for Extended Reality Musical Instruments (Python-Unity Bridge)

This Python project provides a bridge for transmitting sEMG (surface electromyography) data to Unity or other platforms using ZeroMQ. The system is capable of preprocessing sEMG data in real-time and is designed to integrate seamlessly with Unity for various applications.
It is part of the multimodal XR Hand Tracking system for Extended Reality Musical Instruments (XRMIs), as described in our research paper "Combining Vision and EMG-Based Hand Tracking for Extended Reality Musical Instruments".
Included are files for processing, analyzing and transmitting sEMG data using the Myo armband with Python.
The code is part of a research project at Queen Mary University of London, which aims to develop a multimodal hand tracking system for XRMIs.

Note: The Unity implementation for the multimodal hand tracking system can be found [here](https://github.com/maxgraf96/sEMG-myo-unity).

Note 2: The code for training the deep learning model can be found [here](https://github.com/maxgraf96/seMG-myo-python).

## Project Structure

- `data_collection.py`: Collects sEMG data from the Myo armband and transmits it via ZeroMQ.
- `data_processing.py`: Handles the preprocessing of the EMG data.
- `inference.py`: For running inference using the preprocessed data.
- `worker.py`: Manages the ZeroMQ connections for data transmission.
- `data/`: Optionally, EMG data can be saved to this directory.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- ZeroMQ
- [Unity project](https://github.com/maxgraf96/sEMG-myo-unity) (if transmitting data to Unity)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/maxgraf96/sEMG-unity-bridge
   ```
2. Navigate to the project directory:
   ```
   cd sEMG-unity-bridge
   ```
3. Install the necessary Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Collecting and sending sEMG data:

1. Ensure the Myo armband is connected to your computer using the official software.
2. Run the data collection loop:
   ```
   python data_collection.py
   ```

#### Inference (predicting finger joint angles from sEMG data):   
1. Ensure the Myo armband is connected to your computer using the official software.
2. Train the model (see [here](https://github.com/maxgraf96/seMG-myo-python)) or use ours (`model_rnn.onnx`).
3. Run inference:
   ```
   python inference.py
   ```
   This will send the inference results to the Unity project (if running) or print them to the console.

## Contributing

Contributions are welcome. Please feel free to submit pull requests or open issues to suggest improvements or add new features.


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

- Max Graf - [max.graf@qmul.ac.uk](mailto:max.graf@qmul.ac.uk)
- Project Link: [https://github.com/maxgraf96/sEMG-myo-unity](https://github.com/maxgraf96/sEMG-myo-unity)

## Citation
If you use this work, please cite
   ```
   @misc{graf2023combining,
      title={Combining Vision and EMG-Based Hand Tracking for Extended Reality Musical Instruments}, 
      author={Max Graf and Mathieu Barthet},
      year={2023},
      eprint={2307.10203},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
   ```