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

"""
### Demo 2

This example uses some tricks to read and store data in a 2D array.

Expected results:

input  is: 
 tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], device='cuda:0')
output is: 
 tensor([[1., 1., 1., 0.],
        [1., 1., 1., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], device='cuda:0')
        
"""

@triton.jit
def demo2_kernel(input_ptr, output_ptr):
    # Generate the row index range (0 to 7)
    # [0, 1, 2, 3, 4, 5, 6, 7]
    i_range = tl.arange(0, 8)[:, None]
    # Generate the column index range (0 to 4)
    # [ [0], [1], [2], [3] ]
    j_range = tl.arange(0, 4)[None, :]
    # Calculate the range
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15],
    #  [16, 17, 18, 19],
    #  [20, 21, 22, 23],
    #  [24, 25, 26, 27],
    #  [28, 29, 30, 31]]
    range = i_range * 4 + j_range

    # [[ True,  True,  True, False],
    #  [ True,  True,  True, False],
    #  [False, False, False, False],
    #  [False, False, False, False],
    #  [False, False, False, False],
    #  [False, False, False, False],
    #  [False, False, False, False],
    #  [False, False, False, False]]
    mask = (i_range < 2) & (j_range < 3)

    # Load x from DRAM.
    x = tl.load(input_ptr + range, mask, 0)
    # Write x back to DRAM.
    tl.store(output_ptr + range, x, mask)

def demo2():
    input = torch.ones(4, 4)
    output= torch.zeros_like(input)

    print(f'input  is: \n {input}')
    demo2_kernel[(1, 1, 1)](input, output)
    print(f'output is: \n {output}')

"""
### Demo 3

This example illustrates how to write to a tensor.

Expected results:

input  is: 
 tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
output is: 
 tensor([[10., 10., 10.],
        [10., 10.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]], device='cuda:0')

"""

@triton.jit
def demo3_kernel(output_ptr):
    range = tl.arange(0, 8)
    tl.store(output_ptr + range, 10, range < 5)

def demo3():
    output= torch.ones(4, 3)

    print(f'input  is: \n {output}')
    demo3_kernel[(1, 1, 1)](output)
    print(f'output is: \n {output}')

"""

Demo 4

This is an example with one program axis with 3 blocks.

Expected results:

input  is: 
 tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]], device='cuda:0')
output is: 
 tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]], device='cuda:0')

"""

@triton.jit
def demo4_kernel(input_ptr, output_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    mask = range < 20

    x = tl.load(input_ptr + range, mask)
    tl.store(output_ptr + range, x, mask)

def demo4():
    input = torch.ones(2, 4, 4)
    output= torch.zeros_like(input)

    print(f'input  is: \n {input}')
    demo4_kernel[(3, 1, 1)](input, output)
    print(f'output is: \n {output}')
    
if __name__ == '__main__':
    torch.set_default_device('cuda:0')

    # demo1()
    # demo2()
    # demo3()
    demo4()