The codes in this folder mainly focus at tracing the psp trajectory back to the surface of the Sun.
Folder Structure:
<main
      <SC2SolarSurface.py
      <download_data.py
      <data
            <psp
            <gong
                  <adapt
      <res
           <adapt
 
After creating a proper folder, the user can use the code via "python3 SC2SolarSurface.py".
The codes adopt Parker Spiral to trace back to the source surface(the height of which needs be defined by the user), 
then pfss extrapolation from source surface to the surface of the Sun.

The user needs to input the start/end time of PSP, and the time of gong-adapt, the index of the gong-adapt map sequence(0~11), 
the height of the source surface to run the code. (Just run the code! And input the information when the code asks you to!)

The data files of the example are not uploaded. However, there is a data downloading section in the code, try to run the code and download the data on your own! 

Parameters of the example are as follows:
The start/end time of the example:
"2021-04-23T00:00:00"/"2021-05-02T00:00:00".
The adapt-gong time of the example:
"2021-05-01T22:00:00".
The index of the adapt-map sequence:
11.
The height of the source surface:
2.0.

If error happens because of the loss of some packages, please "pip install xx" them.

Enjoy!
