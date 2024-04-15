# libmcraw

#### A simple library for decoding files recorded by [MotionCam Pro](https://www.motioncamapp.com/)

This library is based on the [MotionCam Decoder](https://github.com/mirsadm/motioncam-decoder) published by the author of MotionCam Pro.

## Library

The library is located in the `lib` directory. 

## Build

To build the sample:

```
mkdir build
cd build
cmake ..
make
```

## Usage

In the `samples` directory are two samples demonstrating the basic usage.

To dump the metadata and file structure of a `.mcraw` file run:

`./metadata <path to mcraw file> <dump file structure 1|0>`

To export the audio to a PCM data file:

`./export_audio <path to mcraw file> <path to output PCM file>`
