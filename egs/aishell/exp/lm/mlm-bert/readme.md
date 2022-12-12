### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 102.29
* GPU info \[4\]
  * \[4\] Tesla P100-PCIE-16GB

### Notes

* This is an experiment to fine-tune BERT with the masked language model (MLM) method on aishell. 

### Result
|CER type     | BERT |  BERT after fine-tuning  |
| -------     | -------- | ----------- |
| in-domain   | 3.29     |  3.11       | 
| cross-domain| 3.65     |  3.44       | 


|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|
