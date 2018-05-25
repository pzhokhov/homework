Results for 100 rollouts of training data, 30 epochs of BC agent training, and 100 test 
rollouts are summarized below

|  Env            | Mean reward       | Reward std | Expert mean reward | expert reward std|
|-----------------|-------------------|------------|--------------------|------------------|
| Ant-v2          | 4754.9            | 355.2      | 4772.3             | 214.4            |
| HalfCheetah-v2  | 4118.5            | 107.7      | 4130.2             | 86.3             |
| Hopper-v2       | 1568.0            | 528.6      | 3777.7             | 3.5              |
| Humanoid-v2     | 897.5             |  326.7     | 10408.0            | 52.9             |
| Reacher-v2      | -30.8             | 63.2       | -4.1               | 1.7              |
| Walker2d-v2     | 4281.9            | 2056.1     | 5510.7             | 216.4            |

On the Ant-v2, HalfCheetah the BC agent achieves reward almost as high as the expert (although with slightly higher variance).
On other tasks BC agent performs worse (for Humanoid order of magitude worse) than the expert. 


