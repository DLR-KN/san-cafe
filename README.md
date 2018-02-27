# SAN-CAFE The Coprocessor Accelerated Filterbank Extension Library

## About
CAFE is a CUDA implementation of a Polyphase Filterbank Channelizer and Resampler. It is developed by the Satellite Network Department at the Institute of Communication and Navigation, German Aerospace Centre. 

## Build
With the shell of your choice, navigate into the source file tree.

```
mkdir build
cmake ../
make -j4
```

### Usage
Please refer to the testing folder, to get an impression how to use the library.

### Notes
While both Filters have been tested in DLR research and prototype Software Defined Radio testbeds, consider the status of this software to be still in alpha mode. 

* The Channelizer is still missing a dedicated unit test for example, but this is on our To Do List. 
* The Arbitrary Resampler only supports resampling rates that will generate integer filter skips. So num_filters/rate must be an integer. 

Anyway, if you play around with it and discover bugs or missing features, please do not hesitate to use Githubs Issue function and let us know!
