

### HDC-Approximation Speedup / Accuracy Curve (1)

**Accuracy / QoS**
We use these terms interchangeably, depending on the application it is 'accuracy', or 'mutual information score'

We plot the QoS vs speedup tradeoff for various applications on various backends.


#### HD-Clustering

##### GPU
Speedup over slowest, not baseline. Usually equivalent w/ the currently enabled knobs.
Cropped for accuracy between .4 and .7.
Ran on MNIST

![gpu](./attachments/clustering-250-gpu.png)

##### CPU
Speedup opver slowest, not baseline. 
Ran on MNIST

![cpu](./attachments/clustering-cpu.png)

##### SIM
UCSD folks said that this kinda was possible, but like didn't seem super confident. Might be bugged. 

![zoomed in](./attachments/sim.png)
![image info](./attachments/sim1.png)


### HDC-Approximation Speedup / Tuning-Time (2)
Across different target accuracy constraints, how long does it take (x) to get (y) speedup.

