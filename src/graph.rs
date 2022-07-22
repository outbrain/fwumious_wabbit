
use crate::regressor::BlockTrait;


#[derive(Copy, Clone)]
pub struct BlockOutput(usize);
#[derive(Copy, Clone)]
pub struct BlockInput(usize);
#[derive(Copy, Clone)]
pub struct BlockPtr(usize);	// just an id in a graph
#[derive(Copy, Clone)]
pub struct BlockPtrOutput(BlockPtr, BlockOutput); // since blocks can have multiple outputs, separate between them
#[derive(Copy, Clone)]
pub struct BlockPtrInput(BlockPtr, BlockInput); // since blocks can have multiple inputs, separate between them


pub struct BlockGraph {
    pub blocks: Vec<Box<dyn BlockTrait>>,
    pub edges_in: Vec<Vec<Vec<BlockPtrOutput>>>,	// each block can have multiple input edge groups
    pub edges_out: Vec<Vec<Vec<BlockPtrInput>>>,  // each block can have multiple output edge groups
}



impl BlockGraph {
    pub fn new() -> BlockGraph {
        BlockGraph {blocks: Vec::new(),
                    edges_in: Vec::new(),
                    edges_out: Vec::new()
                    }
    }

    pub fn add_node(&mut self, 
                    mut block: Box<dyn BlockTrait>, 
                    edges_in: Vec<Vec<BlockPtrOutput>>
                    ) -> Vec<BlockPtrOutput> {
        let b = BlockPtr(self.blocks.len());
        self.blocks.push(block);
        for (i, edges_in2) in edges_in.iter().enumerate() {
            let bi = BlockInput(i);
            for e in edges_in2.iter() {
                let bpi = BlockPtrInput(b, bi);
                self.edges_out[e.0.0][e.1.0].push(bpi)
            }
        }
        self.edges_in.push(edges_in);
        let num_output_connectors = 1;        
        let mut vo:Vec<BlockPtrOutput> = Vec::new();
        for i in 0..num_output_connectors {
            let bo = BlockPtrOutput(b, BlockOutput(i));
            vo.push(bo);
            self.edges_out[b.0].push(Vec::new()); // make empty spaceg
        }
        
        return vo;
        
    }
}
