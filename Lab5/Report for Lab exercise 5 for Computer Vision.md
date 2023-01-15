### Report for Lab exercise 5 for Computer Vision

- Distance Function and Gaussian Function

```python
# I found that computing sqrt of sqare sum is faster than using norm function.
def distance(x, X):
    return torch.sqrt((X - x)[:, 0]**2 + (X - x)[:, 1]**2 + (X - x)[:, 2]**2)
# For vectorization version, weight can be calculated by F(exp(dist)), it would return a vector too.
def gaussian(dist, bandwidth):
    weight = 1/bandwidth/np.sqrt(2) * torch.exp(-dist ** 2 / 2 / (bandwidth**2))
    return weight/torch.sum(weight)
```

- Normalization term in gaussian

First time I did not add the normalization term in gaussian function:

Then I found that when updating points, the value of X becomes larger and larger, achieving:<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221122034119856.png" alt="image-20221122034119856" style="zoom:50%;" />

Finally it leads to an error because the values now surpass the boundary and will not converge a steady state

- Update_point Function

```python
    x_update = torch.mm(weight.double(), X.double())
```

- Acceleration

Using for loop in gaussian (vectorization code is above)

```python
    for i in range(len(dist)):
        weight[i] = 1/bandwidth/np.sqrt(2) * torch.exp(-dist[i] ** 2 / 2 / (bandwidth**2))
```

This is so damn slow whatever using gpu or cpu, and my computer runs forever and doesn't get a outcome in hours.

Using Vectorization with cpu<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221122042135161.png" alt="image-20221122042135161" style="zoom: 25%;" />

Using Vectorization with gpu<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221121202543719.png" alt="image-20221121202543719" style="zoom: 25%;" />

In conclusion, it's easy to see vectorized-based mean-shift is way faster than for-loop-based mean-shift.

And it's weird that using gpu is slower than using cpu. It might because that this is a very simple calculation, which means that gpu is as the same as cpu calculation. Gpu is a little slower just because the extra copying the data from cpu to cuda.

- Result<img src="D:\22fall\CV\CV_22Fall\Lab5\mean-shift\mean-shift_cow\result.png" alt="result" style="zoom: 25%;" />