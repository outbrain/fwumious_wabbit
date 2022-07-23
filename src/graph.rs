
use crate::regressor::BlockTrait;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockOutput(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockInput(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtr(usize);	// just an id in a graph
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtrOutput(BlockPtr, BlockOutput); // since blocks can have multiple outputs, separate between them
#[derive(Copy, Clone, Debug, PartialEq)]
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
                    
        let bp = BlockPtr(self.blocks.len());
        self.blocks.push(block);
        for (i, edges_in2) in edges_in.iter().enumerate() {
            let bi = BlockInput(i);
            for e in edges_in2.iter() {
                let bpi = BlockPtrInput(bp, bi);
                self.edges_out[e.0.0][e.1.0].push(bpi)
            }
        }
        self.edges_in.push(edges_in);
        self.edges_out.push(Vec::new());
        let num_output_connectors = 1;        
        let mut vo:Vec<BlockPtrOutput> = Vec::new();
        for i in 0..num_output_connectors {
            let bo = BlockPtrOutput(bp, BlockOutput(i));
            vo.push(bo);
            self.edges_out[bp.0].push(Vec::new()); // make empty spaceg
        }
        
        return vo;        
    }
    
    fn get_num_input_connections(&self, bp: BlockPtr) -> usize {
        self.edges_in[bp.0].len()
    }
    
    fn len(&self) -> usize {
        assert!(self.blocks.len() == self.edges_in.len());
        assert!(self.blocks.len() == self.edges_out.len());
        self.blocks.len()
    }
    
    pub fn to_block_list(self) {
        // we  need to figure out the order
        // first find nodes with no inputs
        let mut output_list: Vec<BlockPtr> = Vec::new();
        let mut remaining_list: Vec<BlockPtr> = Vec::new();
        let mut num_inputs_done: Vec<usize> = Vec::new();
        for i in 0..self.len() {
            remaining_list.push(BlockPtr(i));
            num_inputs_done.push(0);
        }
        
        while remaining_list.len() > 0 {
            for bp in remaining_list.iter() {
                if self.edges_in[bp.0].len() == num_inputs_done[bp.0] {
                    output_list.push(BlockPtr(bp.0));
                }
            }
        }
         
        
                
    }
   
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_loss_functions;
   
    
    #[test]
    fn graph_creation() {
        let mut bg = BlockGraph::new();
        
        let mut ib = block_loss_functions::new_zero_block(1).unwrap();
        let output_nodes = bg.add_node(ib, vec![]);
        assert_eq!(output_nodes, vec![BlockPtrOutput(BlockPtr(0), BlockOutput(0))]);
        assert_eq!(bg.edges_in[0].len(), 0);      // basically []
        assert_eq!(bg.edges_out[0].len(), 1);	  // basically [[]]	
        assert_eq!(bg.edges_out[0][0].len(), 0);  

        let mut ib = block_loss_functions::new_result_block(1, 1.0).unwrap();
        let output_nodes = bg.add_node(ib, vec![output_nodes]);
        assert_eq!(output_nodes, vec![BlockPtrOutput(BlockPtr(1), BlockOutput(0))]);
//        println!("Output nodes: {:?}", output_nodes);
        println!("edges_in: {:?}, edges_out: {:?}", bg.edges_in[0], bg.edges_out[0]);
        println!("edges_in: {:?}, edges_out: {:?}", bg.edges_in[1], bg.edges_out[1]);
    }
}
