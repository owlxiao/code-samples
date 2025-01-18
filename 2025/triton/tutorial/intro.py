import torch 

import triton 
import triton.language as tl

"""
### Demo1 

This is an example of load. It takes an `arange` over the memory.
By default the indexing of torch tensors with column, rows, depths or right-to-left.
It also takes in a mask as the second argument. 
Mask is cirtically important because all shapes in Triton need to be powers of two.

In this demo, we load a 4x3 tensor `x` into the GPU, and write `x[0, 4]` back to DRAM.

Expected results:

input  is: 
 tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
output is: 
 tensor([[1., 1., 1.],
        [1., 1., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], device='cuda:0')
"""
@triton.jit 
def demo1_kernel(input_ptr, output_ptr):
    # Generate a range of indices 
    # [0, 1, 2, 3, 4, 5, 6, 7]
    range = tl.arange(0, 8)
    # Create a mask to access the memory, 
    # we only need the first few data.
    # mask = [1 1 1 1 1 0 0 0]
    mask = range < 5
    
    # Load x from DRAM, masking out extra elements.
    x = tl.load(input_ptr + range, mask, 0)
    # Write x back to DRAM
    # output = [0 1 2 3 4 0 0 0]
    tl.store(output_ptr + range, x, mask)

def demo1():
    print('Demo1 Output:')
    input = torch.ones(4, 3)
    output= torch.zeros_like(input)

    print(f'input  is: \n {input}')
    demo1_kernel[(1, 1, 1)](input, output)
    print(f'output is: \n {output}')
    
if __name__ == '__main__':
    torch.set_default_device('cuda:0')

    demo1()