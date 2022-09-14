The codes in this folder mainly foucu at tracing the psp trajectory back to the surface of the Sun.
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

If error happens because of the loss of some packages, please "pip install xx" them.

Enjoy!
