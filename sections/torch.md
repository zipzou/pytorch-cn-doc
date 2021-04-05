## Torch

torch包含了多位张量数据结构和相关的数学运算，另外，它还提供了大量高效的张量和特定类型的序列化工具以及其他常用工具。
torch含有CUDA版本，使得你可以将张量运算放到Nvidia GPU中。


### 张量

| | |
| --- | --- |
| `is_tensor` | 如果参数是PyTorch张量时返回`True`，否则`False` |
| `is_storage` | 如果参数是PyTorch存储对象时，返回`True`，否则`False`|
| `is_complex` | 如果输入的数据类型是一个复数类型时返回`True`，即数据类型是:[`torch.complex64`]()和[`torch.complex128`]()。 |
|`is_floating_point`|如果输入的数据类型时一个浮点类型返回`True`，即张量的数据类型是：[`torch.float64`]()、[`torch.float32`]()、[`torch.float16`]()和[`torch.bfloat16`]()。|
|`is_nonzero` | 如果输入的张量中每个元素经过类型转换后都为非0值时返回`True`。|
|`set_default_dtype`|将浮点张量的默认数据类型`dtype`设置为`d`。|
|`get_default_dtype`|获取当前浮点张量的默认数据类型，即`torch.dtype`。|
|`set_default_tensor_type`|将`torch.Tensor`默认浮点类型设置为类型`t`。|
|`numel`|返回输入张量中的所有元素的数量。|
|`set_printoptions`|设置打印选项。|
|`set_flush_denormal`|在CPU上禁用非标准浮点数。|

### 张量创建

备注：

随机采样创建算子包含如下随机采样方法：`torch.rand()`，`torch.rand_like()`，`torch.randn()` `torch.randn_like()`，`torch.randint()`，`torch.randint_like()`，`torch.randperm()` 你也可以使用`torch.empty()`来使用In-place随机方法来创建`torch.Tensor`张量，得到更广泛的随机采样值。

| | |
| --- | --- |
|`tensor`|使用指定数据创建一个张量。|
|`sparse_coo_tensor`|使用给定的索引和值来创建一个[坐标格式的稀疏张量]()。|
|`as_tensor`|将数据转化为`torch.Tensor`类型|
|`as_strided`|使用指定`size`、`stride`和`storage_size`来创建已有`torch.Tensor`的一个视图。|
|`from_numpy`|从`numpy.ndarray`数据中创建张量。|
|`zeros`|返回一个由标量值0填充的张量，张量形状由参数`size`确定。|
|`zeros_like`|返回一个由标量值0填充的张量，张量形状同`input`参数相同。|
|`ones`|返回一个由标量值1填充的张量，张量形状由参数`size`确定。|
|`ones_like`|返回一个由标量值1填充的张量，张量形状同`input`参数相同。|
|`arange`|返回一个大小为 $\left\lceil \frac{end-start}{step} \right\rceil$的张量，其初始值在$[start,end)$范围取得，并且每个值的差值为`step`。|
|`range`|返回一个大小为$\left\lfloor\frac{end-start}{step}\right\rfloor+1$的张量，其初始值在$[start,end)$范围取得，每个值间隔为`step`。|
|`linspace`|创建一个大小为`steps`的一维张量，其值均匀地从`start`到`end`中采样。|
|`logspace`|创建一个大小为`steps`的一维张量，其值均匀地从$base^{start}$到$base^{end}$中采样取得。|
`eye`|返回一个对角线为1其他为0的2维张量。|
|`empty`|返回一个无初始值填充的张量。|
`empty_like`|返回一个无初始值填充的张量，其形状同输入`input`一致。|
|`empty_strided`|返回一个无初始值填充的张量。|
`full`|返回一个大小为`size`且初始值由`fill_value`填充的张量。|
|`full_like`|返回一个大小同`input`一致，初始值由`fill_value`填充的张量。|
|`quantize_per_tensor`|将一个浮点张量转化为由给定值和0点的四分位张量。|
|`quantize_per_channel`|将一个浮点张量转化为由给定值和0点的按通道四分的张量。|
|`dequantize`|对一个四分张量取消四分，返回一个数据类型为fp32的张量。|
|`complex`|根据实部`real`和虚部`imag`创建一个复数张量。|
|`polar`|创建一个复数张量，其元素是绝对值`abs`和角度`angle`对应极坐标所表示的笛卡尔坐标。|
|`heaviside`|对`input`中的每个元素计算`Heaviside`步长变换。|

### 索引、切片、连接和变形操作

| | |
| --- | --- |
|`cat`|将给定张量序列`seq`按指定维度拼接。|
|`chunk`|将张量拆分为特定数量的块。|
|`column_stack`|通过水平堆叠张量`tensors`中的张量来创建一个新的张量|
|`dstack`|Stack tensors in sequence depthwise (along third axis).|
|`gather`|通过给定维度`dim`的轴来获取值。|
|`hstack`|在序列水平方向上堆叠张量(按列堆叠)。|
|`index_select`|根据维度`dim`和`index`中的索引，选取对应元素返回一个新的张量，`index`为`LongTensor`。|
|`masked_select`|根据布尔掩码`mask`(`BoolTensor`)索引输入张量`input`，并返回一个新的1维张量。|
|`movedim`|将输入张量`input`的`source`中的维度位置移动到`destination`位置。｜
|`moveaxis`|`torch.movedim()`的别名。|
|`narrow`|返回输入张量`input`的压缩版本。|
|`nonzero`| |
|`reshape`|返回指定形状且同`input`张量数据和元素数量相同的新张量。|
|`row_stack`|`torch.vstack()`的别名。|
|`scatter`|`torch.Tensor.scatter_()`的Out-of-place版。|
|`scatter_add`|`torch.Tensor.scatter_add_()`的别名。|
|`split`|将张量分块。|
|`squeeze`|将`input`张量中维度值为1的维度移除，返回移除后的新张量。|
|`stack`|按照新的维度将张量序列拼接。|
|`swapaxes`|`torch.transpose()`的别名。|
|`swapdims`|`torch.transpose()`的别名。|
|`t`|需确保`input`维度$\leq 2$维，并且转置第0和1维。|
|`take`|返回由`input`中给定索引对应元素构成的新张量。|
|`tensor_split`|将张量拆分为多个子张量，Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.|
|`tile`|通过重复`input`中的元素构建张量。|
|`transpose`|返回`input`的转置张量。|
|`unbind`|移除张量的维度。|
|`unsqueeze`|返回在指定位置插入维度值为1后的新张量。|
|`vstack`|竖直地堆叠张量（按行）。|
|`where`|返回一个值为`x`或`y`的新张量，当条件`condition`成立时为`x`，否则为`y`。|

### 生成器


| | |
| --- | --- |
|`Generator`|Creates and returns a generator object that manages the state of the algorithm which produces pseudo random numbers.|

### 随机采样

| | |
| --- | --- |
|`seed`|Sets the seed for generating random numbers to a non-deterministic random number.|
|`manual_seed`|Sets the seed for generating random numbers.|
|`initial_seed`|Returns the initial seed for generating random numbers as a Python long.|
|`get_rng_state`|Returns the random number generator state as a torch.ByteTensor.
|`set_rng_state`|Sets the random number generator state.|


### `torch.default_generator Returns the default CPU torch.Generator`


| | |
| --- | --- |
|`bernoulli`|Draws binary random numbers (0 or 1) from a Bernoulli distribution.
|`multinomial`|Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.|
|`normal`|Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.|
|`poisson`|Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input i.e.,|
|`rand`|Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)[0,1)|
|`rand_like`|Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1)[0,1) .|
|`randint`|Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).|
|`randint_like`|Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).|
|`randn`|Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).|
|`randn_like`|Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.|
|`randperm`|Returns a random permutation of integers from 0 to n - 1.|

### In-place随机采样

There are a few more in-place random sampling functions defined on Tensors as well. Click through to refer to their documentation:

+ `torch.Tensor.bernoulli_()` - in-place version of torch.bernoulli()
+ `torch.Tensor.cauchy_()` - numbers drawn from the Cauchy distribution
+ `torch.Tensor.exponential_()` - numbers drawn from the exponential distribution
+ `torch.Tensor.geometric_()` - elements drawn from the geometric distribution
+ `torch.Tensor.log_normal_()` - samples from the log-normal distribution
+ `torch.Tensor.normal_()` - in-place version of torch.normal()
+ `torch.Tensor.random_()` - numbers sampled from the discrete uniform distribution
+ `torch.Tensor.uniform_()` - numbers sampled from the continuous uniform distribution

### Quasi-随机采样

| | |
| --- | --- |
|`quasirandom.SobolEngine`|The torch.quasirandom.SobolEngine is an engine for generating (scrambled) Sobol sequences.|

### 序列化


| | |
| --- | --- |
|`save`|Saves an object to a disk file.|
|`load`|Loads an object saved with torch.save() from a file.|

### 并行化

| | |
| --- | --- |
|`get_num_threads`|Returns the number of threads used for parallelizing CPU operations|
|`set_num_threads`|Sets the number of threads used for intraop parallelism on CPU.|
|`get_num_interop_threads`|Returns the number of threads used for inter-op parallelism on CPU (e.g.|
|`set_num_interop_threads`|Sets the number of threads used for interop parallelism (e.g.|

### 本地禁用梯度计算

The context managers `torch.no_grad()`, `torch.enable_grad()`, `and torch.set_grad_enabled()` are helpful for locally disabling and enabling gradient computation. See Locally disabling gradient computation for more details on their usage. These context managers are thread local, so they won’t work if you send work to another thread using the threading module, etc.

示例:
```:bash
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```

| | |
| --- | --- |
|`no_grad`|禁用梯度计算的上下文管理器。|
|`enable_grad`|启用梯度计算的上下文管理器。|
|`set_grad_enabled`|设置上下文管理器是否开启梯度计算。|

### 数学运算

#### 浮点运算

| | |
| --- | --- |
|`abs`|计算`input`每个元素的绝对值。|
|`absolute`|`torch.abs()`的别名。|
|`acos`|计算`input`每个元素的反余弦值。|
|`arccos`|`torch.acos()`的别名。|
|`acosh`|计算`input`每个元素的反双曲余弦值。|
|`arccosh`|`torch.acosh()`的别名。|
|`add`|Adds the scalar other to each element of the `input` input and returns a new resulting tensor.|
|`addcdiv`|Performs the element-wise division of tensor1 by tensor2, multiply the result by the |scalar value and add it to input.
`addcmul`|Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
|`angle`|Computes the element-wise angle (in radians) of the given input tensor.|
|`asin`|Returns a new tensor with the arcsine of the elements of input.|
|`arcsin`|Alias for torch.asin().|
|`asinh`|Returns a new tensor with the inverse hyperbolic sine of the elements of input.|
|`arcsinh`|Alias for torch.asinh().|
|`atan`|Returns a new tensor with the arctangent of the elements of input.|
|`arctan`|Alias for torch.atan().|
|`atanh`|Returns a new tensor with the inverse hyperbolic tangent of the elements of input.|
|`arctanh`|Alias for torch.atanh().|
|`atan2`|Element-wise arctangent of input i/other i input  i/other i with consideration of the quadrant.|
|`bitwise_not`|Computes the bitwise NOT of the given input tensor.|
|`bitwise_and`|Computes the bitwise AND of input and other.|
|`bitwise_or`|Computes the bitwise OR of input and other.|
|`bitwise_xor`|Computes the bitwise XOR of input and other.|
|`ceil`|Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.|
|`clamp`|Clamp all elements in input into the range [ min, max ].|
|`clip`|Alias for torch.clamp().|
|`conj`|Computes the element-wise conjugate of the given input tensor.|
|`copysign`|Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.|
|`cos`|Returns a new tensor with the cosine of the elements of input.|
|`cosh`|Returns a new tensor with the hyperbolic cosine of the elements of input.|
|`deg2rad`|Returns a new tensor with each of the elements of input converted from angles in degrees to radians.|
|`div`|Divides each element of the input input by the corresponding element of other.|
|`divide`|Alias for torch.div().|
|`digamma`|Computes the logarithmic derivative of the gamma function on input.|
|`erf`|Computes the error function of each element.|
|`erfc`|Computes the complementary error function of each element of input.|
|`erfinv`|Computes the inverse error function of each element of input.|
|`exp`|Returns a new tensor with the exponential of the elements of the input tensor input.|
|`exp2`|Computes the base two exponential function of input.|
|`expm1`|Returns a new tensor with the exponential of the elements minus 1 of input.|
|`fake_quantize_per_channel_affine`|Returns a new tensor with the data in input fake quantized per channel using scale, zero_point, quant_min and quant_max, across the channel specified by axis.|
|`fake_quantize_per_tensor_affine`|Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min and quant_max.|
|`fix`|Alias for torch.trunc()|
|`float_power`|Raises input to the power of exponent, elementwise, in double precision.|
|`floor`|Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.|
|`floor_divide`| |
`fmod`|Computes the element-wise remainder of division.|
|`frac`|Computes the fractional portion of each element in input.|
|`imag`|Returns a new tensor containing imaginary values of the self tensor.|
|`ldexp`|Multiplies input by 2**:attr:other.|
|`lerp`|Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.|
|`lgamma`|Computes the logarithm of the gamma function on input.|
|`log`|Returns a new tensor with the natural logarithm of the elements of input.|
|`log10`|Returns a new tensor with the logarithm to the base 10 of the elements of input.|
|`log1p`|Returns a new tensor with the natural logarithm of (1 + input).|
|`log2`|Returns a new tensor with the logarithm to the base 2 of the elements of input.|
|`logaddexp`|Logarithm of the sum of exponentiations of the inputs.|
|`logaddexp2`|Logarithm of the sum of exponentiations of the inputs in base-2.|
|`logical_and`|Computes the element-wise logical AND of the given input tensors.|
|`logical_not`|Computes the element-wise logical NOT of the given input tensor.|
|`logical_or`|Computes the element-wise logical OR of the given input tensors.|
|`logical_xor`|Computes the element-wise logical XOR of the given input tensors.|
|`logit`|Returns a new tensor with the logit of the elements of input.|
|`hypot`|Given the legs of a right triangle, return its hypotenuse.|
|`i0`|Computes the zeroth order modified Bessel function of the first kind for each element of input.|
|`igamma`|Computes the regularized lower incomplete gamma function:|
|`igammac`|Computes the regularized upper incomplete gamma function:|
|`mul`|Multiplies each element of the input input with the scalar other and returns a new resulting tensor.|
|`multiply`|Alias for torch.mul().|
|`mvlgamma`|Computes the multivariate log-gamma function) with dimension 
|`p`|p element-wise, given by|
|`nan_to_num`|Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.|
|`neg`|Returns a new tensor with the negative of the elements of input.|
|`negative`|Alias for torch.neg()|
|`nextafter`|Return the next floating-point value after input towards other, elementwise.
|`polygamma`|Computes the nth n th derivative of the digamma function on input.|
|`pow`|Takes the power of each element in input with exponent and returns a tensor with the result.|
|`rad2deg`|Returns a new tensor with each of the elements of input converted from angles in radians to degrees.|
|`real`|Returns a new tensor containing real values of the self tensor.|
|`reciprocal`|Returns a new tensor with the reciprocal of the elements of input|
|`remainder`|Computes the element-wise remainder of division.|
|`round`|Returns a new tensor with each of the elements of input rounded to the closest integer.|
|`rsqrt`|Returns a new tensor with the reciprocal of the square-root of each of the elements of input.|
|`sigmoid`|Returns a new tensor with the sigmoid of the elements of input.|
|`sign`|Returns a new tensor with the signs of the elements of input.|
|`sgn`|For complex tensors, this function returns a new tensor whose elemants have the same angle as that of the elements of input and absolute value 1.|
|`signbit`|Tests if each element of input has its sign bit set (is less than zero) or not.
|`sin`|Returns a new tensor with the sine of the elements of input.|
|`sinc`|Computes the normalized sinc of input.|
|`sinh`|Returns a new tensor with the hyperbolic sine of the elements of input.|
|`sqrt`|Returns a new tensor with the square-root of the elements of input.|
|`square`|Returns a new tensor with the square of the elements of input.|
|`sub`|Subtracts other, scaled by alpha, from input.|
|`subtract`|Alias for torch.sub().|
|`tan`|Returns a new tensor with the tangent of the elements of input.|
|`tanh`|Returns a new tensor with the hyperbolic tangent of the elements of input.|
|`true_divide`|Alias for torch.div() with rounding_mode=None.|
|`trunc`|Returns a new tensor with the truncated integer values of the elements of input.|
|`xlogy`|Computes input * log(other) with the following cases.|


#### Reduction Ops


#### Comparison Ops


#### Spectral Ops


#### Other Operations

### Utilities