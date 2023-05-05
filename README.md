# This project work is done as a part of class project for COMS 576-Motion strategy and its application.

In order to run the project, please navigate to folder in the command line promt, and run specific command for the algorithm you want to run. 


To run the original version of RRT
```
$ python project.py --alg rrt
``` 
To run the PRM 

```
$ python project.py --alg prm 
``` 

To run rrt star with dynamic radius value-
```
$ python project.py --alg rrt_star --type r 
``` 
To run rrt star with dynamic k-nearest neighbors-
```
$ python project.py --alg rrt_star --type k 
```
To run prm star with dynamic radius value-
```
$ python project.py --alg prm_star --type r 
``` 
To run prm star with dynamic k-nearest neighbors-
```
$ python project.py --alg prm_star --type k 
``` 