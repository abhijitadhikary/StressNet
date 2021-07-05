# Description

This is the source code for the paper——"A Deep Neural Network based on Dynamic Dropout Layers for Stress Recognition".

For running this source code. You must have installed Python 3, Pytorch 1.6.0 , Numpy, FFMPEG and Opencv on your platform. You should download the video dataset from the [link (Google Drive)](https://drive.google.com/file/d/1NFUV3wO0i3drY3O7wbVjtPQ4-6W0g0jB/view?usp=sharing) and unzip it into the `videos` directory.





# Requirements

- python>=3.6
- pytorch>=1.6.0
- FFMPEG
- Opencv3-Python
- numpy
- matplotlib



# Directory Branch

```
├── Readme.md                 
├── videos           // you should put video files in              
├── code                     
    ├── dataset.py
    ├── initialization.py       
    ├── main.py        
    ├── models.py                       
    └── train.py              
```



# Run

**First, you location directory  should be 'StressNet'**;

**Make sure put videos on the 'videos' directory**

Run the main.py file.

```
python main().py --(parameters)
```

parameters:

    parser.add_argument('--exp_id', type=str, default='training id')
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--truth', type=int, default=6, help='num of select labels')
P.S: 

(1) For the 'truth' parameter, we highly recommend you choose 6 or 10. If you choose 20, it will cost a long time in data preprocessing.

(2) If you wanna re-train the model under different settings, you should delete the generated 'video_segments' directory and 'dataset' directory and then run the 'main.py'.

(3) Good luck. If you have any problem, feel free to email me.
