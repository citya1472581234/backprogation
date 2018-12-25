# backpropagation
![](https://github.com/citya1472581234/backprogation/blob/master/DNN/picture/1.PNG?raw=true)

2. BN
* 對於第一層來說，將數據歸一化，可能會有較好的結果。
* 對於其他層來說，梯度較不容易消失，例如:梯度都在負的部分，在relu後可能就會消失。
* 有效減緩over-fitting
* 減少不好的初始化影響
* 可以用大一點的 learning rate
* 用在 activation function 前面
* spatial BN 是根據channel 做 BN
![](https://i.imgur.com/aeOfyyB.png)
![](https://i.imgur.com/eoXtFo3.png)


3.Dropout
* 訓練時以概率 P，保留神經元，測試時所有神經元都參與。

4.conv
* dout 對應區域，和卷積和W反轉卷積，等效於dout的每一個值和卷積和相乘，然後對dx對應區進行疊加

![](https://i.imgur.com/wmBts1R.png)


5.optimizer
* 各種 optimizer 方法 https://blog.csdn.net/u014595019/article/details/52989301 
* adam RMS momentum 比較 https://blog.csdn.net/willduan1/article/details/78070086
* Nesterov accelerated gradient (NAG) https://blog.csdn.net/tsyccnh/article/details/76673073
![](https://i.imgur.com/7avHxqq.png)
