# LiquidRecO

Reconstruction library for a 3D [LiquidO](https://liquido.ijclab.in2p3.fr/overview/) detector.

## Installation
Git clone this repository then can be installed using pip:

```
git clone git@github.com:ewanwm/LiquidRecO.git

pip install ./LiquidRecO/
```

## Usage 

After installing, you can use the liquidreco app to run the reconstruction:

```
liquidreco -i <file containing hits>
```

This app has many options and is highly configurable. To see the varius options you can do 

```
liquidreco -h
```

## Example

Initial fiber hits:
![Example event fiber hits](images/example_neut_event_fibers.png)

Are cleaned up and turned into 3 dimensional hits:
![gif](images/example_neut_event.gif)

We can then use one of a number of different reconstruction algorithms on these. One of the most simple (and most effective) is a [Hough transform](https://en.wikipedia.org/wiki/Hough_transform) (implemented in the reconstruction.HoughTransform class):

![Hough tracks](images/example_neut_event_hough.png)

(Note that the hough transform is performed in 3D then the result projected back down to 2D for easier visualisation)