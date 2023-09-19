The codes in this folder mainly focus at tracing the psp trajectory back to the surface of the Sun.
Folder Structure:  
- main  
  - SC2SolarSurface.py  
  - download_data.py
  - data
    - psp
      - sweap
    - gong
      - adapt
    - aia
  - res
    - source_region
 
After creating a proper folder, the user can use the code via "python3 SC2SolarSurface.py".
The codes adopt Parker Spiral to trace back to the source surface(the height of which needs be defined by the user), 
then pfss extrapolation from source surface to the surface of the Sun.

The user needs to input the start/end time of PSP, and the time of gong-adapt, the index of the gong-adapt map sequence(0~11), 
the height of the source surface to run the code. (All the parameters are set in the main() function.)

The data files of the example are not uploaded. However, there is a data downloading section in the code, try to run the code and download the data on your own! 

If the server refuses the request for retrieving data (such as Max retrieve exceeded etc.), please download data manually to the corresponding folder. The code will work.

This code has done the magnetic connectivity part in arXiv:2301.05829.

Enjoy!
