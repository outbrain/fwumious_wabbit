 
use crate::regressor::BlockTrait;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockOutput(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockInput(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtr(usize);	// just an id in a graph
#[derive(Debug, PartialEq)]
pub struct BlockPtrOutput(BlockPtr, BlockOutput); // since blocks can have multiple outputs, separate between them
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtrInput(BlockPtr, BlockInput); // since blocks can have multiple inputs, separate between them

pub struct BlockGraphNode {
    pub block: Box<dyn BlockTrait>,
    pub edges_in: Vec<BlockPtrOutput>,	// each block can have multiple input edges
    pub edges_out: Vec<BlockPtrInput>,  // each block can have multiple output edges

}

pub struct BlockGraph {
    pub nodes: Vec<BlockGraphNode>,
}


impl BlockPtr {
    pub fn get_node_id(&self) -> usize {self.0}
}

impl BlockOutput {
    pub fn get_output_id(&self) -> usize {self.0}
}

impl BlockInput {
    pub fn get_input_id(&self) -> usize {self.0}
}

impl BlockPtrOutput {
    pub fn get_node_id(&self) -> usize {self.0.get_node_id()}
    pub fn get_output_id(&self) -> usize {self.1.get_output_id()}
    pub fn get_output(&self) -> BlockOutput {self.1}
}

impl BlockPtrInput {
    pub fn get_node_id(&self) -> usize {self.0.get_node_id()}
    pub fn get_input_id(&self) -> usize {self.1.get_input_id()}
    pub fn get_input(&self) -> BlockInput {self.1}
}

const BLOCK_PTR_INPUT_DEFAULT:BlockPtrInput = BlockPtrInput(BlockPtr(usize::MAX), BlockInput(usize::MAX));

impl BlockGraph {
    pub fn new() -> BlockGraph {
        BlockGraph {nodes: Vec::new()}
    }

    pub fn add_node(&mut self, 
                    mut block: Box<dyn BlockTrait>, 
                    edges_in: Vec<BlockPtrOutput>
                    ) -> Vec<BlockPtrOutput> {
                    
        let num_output_connectors = block.get_num_output_tapes();
//        let num_input_connectors = block.get_num_input_tapes();
        let bp = BlockPtr(self.nodes.len());     // id of current node
        for (i, e) in edges_in.iter().enumerate() {
            let bi = BlockInput(i);
            let bpi = BlockPtrInput(bp, bi);
            self.nodes[e.get_node_id()].edges_out[e.get_output_id()] = bpi;
        }
        let mut newnode = BlockGraphNode {
                            block: block,
                            edges_in: edges_in,
                            edges_out: Vec::new(),
                        };


        self.nodes.push(newnode);
        let mut vo:Vec<BlockPtrOutput> = Vec::new();
        for i in 0..num_output_connectors {
            let bo = BlockPtrOutput(bp, BlockOutput(i));
            vo.push(bo);
            self.nodes[bp.get_node_id()].edges_out.push(BLOCK_PTR_INPUT_DEFAULT); // make empty spaceg
        }
        return vo;        
    }
    
    
    
    
    pub fn get_num_input_connections(&self, bp: BlockPtr) -> usize {
        self.nodes[bp.get_node_id()].edges_in.len()
    }
    
    pub fn get_num_outputs(&self, outputs: Vec<&BlockPtrOutput>) -> u32 {
        let mut t = 0;
        for x in outputs {
            let node = &self.nodes[x.get_node_id()];
            t += node.block.get_num_outputs2(x.get_output());
        }
        t
    }


    fn len(&self) -> usize {
        self.nodes.len()
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
                if self.nodes[bp.get_node_id()].edges_in.len() == num_inputs_done[bp.0] {
                    output_list.push(BlockPtr(bp.0));
                }
            }
        }
         
        
                
    }
   
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_misc;
   
    
    #[test]
    fn graph_creation() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), BlockOutput(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);  
    }
    
    #[test]
    fn graph_one_sink() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), BlockOutput(0)));
        assert_eq!(bg.nodes[0].edges_in.len(), 0);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);  

        // Let's add one result block 
        let mut output_node = block_misc::new_result_block2(&mut bg, const_block_output, 1.0).unwrap();
        assert_eq!(output_node, ());
//        println!("Output nodes: {:?}", output_nodes);
        //println!("Block 0: edges_in: {:?}, edges_out: {:?}", bg.edges_in[0], bg.edges_out[0]);
        assert_eq!(bg.nodes[0].edges_in, vec![]);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), BlockInput(0))]);  

        assert_eq!(bg.nodes[1].edges_in, vec![BlockPtrOutput(BlockPtr(0), BlockOutput(0))]);
        assert_eq!(bg.nodes[1].edges_out, vec![]);  
    }
    
    #[test]
    fn graph_two_sinks() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), BlockOutput(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // We need to add a copy block to have two sinks
        let mut copies = block_misc::new_copy_block2(&mut bg, const_block_output).unwrap();
        let c2 = copies.pop().unwrap();
        let c1 = copies.pop().unwrap();
        
        
        // Let's add one result block 
        let mut output_node = block_misc::new_result_block2(&mut bg, c1, 1.0).unwrap();
        assert_eq!(output_node, ());
//        println!("Output nodes: {:?}", output_nodes);
        //println!("Block 0: edges_in: {:?}, edges_out: {:?}", bg.edges_in[0], bg.edges_out[0]);
        assert_eq!(bg.nodes[0].edges_in.len(), 0);      
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), BlockInput(0))]);	  	

        // Let's add second result block to see what happens

        let mut output_node = block_misc::new_result_block2(&mut bg, c2, 1.0).unwrap();
        assert_eq!(output_node, ());
        assert_eq!(bg.nodes[0].edges_in, vec![]);      
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), BlockInput(0))]);
        // copy block
        assert_eq!(bg.nodes[1].edges_in, vec![BlockPtrOutput(BlockPtr(0), BlockOutput(0))]);
        assert_eq!(bg.nodes[1].edges_out, vec![BlockPtrInput(BlockPtr(2), BlockInput(0)), BlockPtrInput(BlockPtr(3), BlockInput(0))]);
        // result block 1
        assert_eq!(bg.nodes[2].edges_in, vec![BlockPtrOutput(BlockPtr(1), BlockOutput(0))]);
        assert_eq!(bg.nodes[2].edges_out, vec![]);
        // result bock 2
        assert_eq!(bg.nodes[3].edges_in, vec![BlockPtrOutput(BlockPtr(1), BlockOutput(1))]);
        assert_eq!(bg.nodes[3].edges_out, vec![]);
    }
    
    
    
/*    #[test]
    fn graph_two_sources() {
        let mut bg = BlockGraph::new();
        
        let const_block_output1 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let const_block_output2 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();

        // Let's add one copy block that takes two inputs
        let mut output_node = block_misc::new_copy_block2(&mut bg, vec![const_block_output1, const_block_output2]).unwrap();
        //assert_eq!(output_node, ());
        assert_eq(bg.nodes[2].block

    }    */
    
    
}
