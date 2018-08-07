# Kidnapped-Vehicle-Particle-Filter
 Classwork for the Udacity Self Driving Car Nanodegree
 
 This repository contains an implementation of a particle filter, a form of localization often used by self-driving cars.
 
 ---
 ## Particle Filter Implementation
 
* This particle filter uses 75 particles. I found that using 75 particles allowed the filter to acheive good accuracy, while still having low computational expense.
* To prevent division by zero, I added non-zero tests in a couple places.
## Running the Filter

To run the filter, clone this repository and in `/Kidnapped-Vehicle-Particle-Filter`, run 

`./clean.sh` 

`./build.sh`

`./run.sh`
