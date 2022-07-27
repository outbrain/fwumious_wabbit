 
use crate::regressor::BlockTrait;
use crate::model_instance;
use crate::port_buffer;
use std::mem;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OutputSlot(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InputSlot(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtr(usize);	// just an id in a graph
#[derive(Debug, PartialEq)]
pub struct BlockPtrOutput(BlockPtr, OutputSlot); // since blocks can have multiple outputs, separate between them
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtrInput(BlockPtr, InputSlot); // since blocks can have multiple inputs, separate between them

#[derive(Debug)]
pub struct BlockGraphNode {
    pub edges_in: Vec<BlockPtrOutput>,	// each block can have multiple input edges
    pub edges_out: Vec<BlockPtrInput>,  // each block can have multiple output edges

}

pub struct BlockGraph {
    pub nodes: Vec<BlockGraphNode>,
    pub blocks: Vec<Box<dyn BlockTrait>>,
    pub tape_size: usize,
}


// We need to treat join type in a special way - all inputs need to be consequtive
#[derive(PartialEq, Debug)]
pub enum BlockType {
    Regular,
    Join,
}

impl BlockPtr {
    pub fn get_node_id(&self) -> usize {self.0}
}

impl OutputSlot {
    pub fn get_output_index(&self) -> usize {self.0}
}

impl InputSlot {
    pub fn get_input_index(&self) -> usize {self.0}
}

impl BlockPtrOutput {
    pub fn get_block_ptr(&self) -> BlockPtr {self.0}
    pub fn get_node_id(&self) -> usize {self.0.get_node_id()}
    pub fn get_output_index(&self) -> usize {self.1.get_output_index()}
    pub fn get_output(&self) -> OutputSlot {self.1}
}

impl BlockPtrInput {
    pub fn get_block_ptr(&self) -> BlockPtr {self.0}
    pub fn get_node_id(&self) -> usize {self.0.get_node_id()}
    pub fn get_input_index(&self) -> usize {self.1.get_input_index()}
    pub fn get_input(&self) -> InputSlot {self.1}
}

const BLOCK_PTR_INPUT_DEFAULT:BlockPtrInput = BlockPtrInput(BlockPtr(usize::MAX), InputSlot(usize::MAX));

impl BlockGraph {
    pub fn new() -> BlockGraph {
        BlockGraph {nodes: Vec::new(),
                    blocks: Vec::new(),
                    tape_size: 0}
    }

    pub fn add_node(&mut self, 
                    mut block: Box<dyn BlockTrait>, 
                    edges_in: Vec<BlockPtrOutput>
                    ) -> Vec<BlockPtrOutput> {
                    
        let num_output_connectors = block.get_num_output_slots();
//        let num_input_connectors = block.get_num_input_tapes();
        let bp = BlockPtr(self.nodes.len());     // id of current node
        for (i, e) in edges_in.iter().enumerate() {
            let bi = InputSlot(i);
            let bpi = BlockPtrInput(bp, bi);
            self.nodes[e.get_node_id()].edges_out[e.get_output_index()] = bpi;
        }
        let mut newnode = BlockGraphNode {
                            edges_in: edges_in,
                            edges_out: Vec::new(),
                        };


        self.nodes.push(newnode);
        self.blocks.push(block);
        let mut vo:Vec<BlockPtrOutput> = Vec::new();
        for i in 0..num_output_connectors {
            let bo = BlockPtrOutput(bp, OutputSlot(i));
            vo.push(bo);
            self.nodes[bp.get_node_id()].edges_out.push(BLOCK_PTR_INPUT_DEFAULT); // make empty spaceg
        }
        return vo;        
    }
    
    
    pub fn get_tape_size(&self) -> usize {
        self.tape_size
    }

    pub fn new_port_buffer(&self) -> port_buffer::PortBuffer {
        port_buffer::PortBuffer::new(self.get_tape_size())
    }
    pub fn get_num_input_slots(&self, bp: BlockPtr) -> usize {
        self.nodes[bp.get_node_id()].edges_in.len()
    }
    
    pub fn get_num_output_values(&self, outputs: Vec<&BlockPtrOutput>) -> usize {
        let mut t = 0;
        for x in outputs {
            t += self.blocks[x.get_node_id()].get_num_output_values(x.get_output());
        }
        t
    }


    fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        for i in 0..self.len() { 
            self.blocks[i].allocate_and_init_weights(&mi);
        }
    }
    
    pub fn take_blocks(&mut self) -> Vec<Box<dyn BlockTrait>> {
        mem::take(&mut self.blocks)
    
    }
    
    pub fn schedule(&mut self) {
        let mut offset: usize = 0;
        // We allocate offsets based on input tapes. 
        // This assures that for each block, inputs are placed conesquitvely are consequtive
        // Which is importnat for the JoinBlock that has that requirement
        for i in 0..self.len() {
            for (input_index, edge_in) in self.nodes[i].edges_in.iter().enumerate() {
                let bo = edge_in.get_output();
                let bptr = edge_in.get_node_id();
                let output_len = self.blocks[bptr].get_num_output_values(bo);
//                println!("Block: {}, output: {:?},  output offset: {} ouptut_len: {}", i, bo, offset, output_len);
                let input_block_type = self.blocks[bptr].get_block_type();
                if input_block_type  == BlockType::Regular {
                    self.blocks[bptr].set_output_offset(bo, offset);
                    self.blocks[i].set_input_offset(InputSlot(input_index), offset);
                    offset += output_len as usize; 
                } else if input_block_type == BlockType::Join {
                    // we are special casing Join block
                    // It is zero-copy joining of inputs, which means inputs and outputs share exactly the same space
                    let fake_offset = self.blocks[bptr].get_input_offset(InputSlot(0)).unwrap();
                    self.blocks[bptr].set_output_offset(bo, fake_offset);
                    self.blocks[i].set_input_offset(InputSlot(input_index), fake_offset);
                } else {
                    panic!("Type of block not supported in scheduling: {:?}", input_block_type);
                }
                
            }
        }    
        // now allocate for dead-end outputs
        for i in 0..self.len() {
            for (output_index, edge_out) in self.nodes[i].edges_out.iter().enumerate() {
                if *edge_out == BLOCK_PTR_INPUT_DEFAULT {
                    let bo = OutputSlot(output_index);
                    let output_len = self.blocks[i].get_num_output_values(bo);
 //                   println!("setting block {}, output {:?} to offset {}", i, bo, offset);
                    self.blocks[i].set_output_offset(bo, offset);
                    offset += output_len as usize;
        
                }
            }
        }
        
  //      println!("Nodes: {:?}", self.nodes);     
        /*for i in 0..self.len() {
            let num_output_connectors = self.blocks[i].get_num_output_slots();
            for j in 0..num_output_connectors {
                let bo = OutputSlot(j);
                let output_len = self.blocks[i].get_num_output_values(bo);
                println!("Block: {}, output: {:?},  output offset: {} ouptut_len: {}", i, bo, offset, output_len);
                self.blocks[i].set_output_offset(bo, offset);
                if self.nodes[i].edges_out[j].get_node_id() != usize::MAX {
                    // you can have dangling outputs... why not
                    self.blocks[self.nodes[i].edges_out[j].get_node_id()].set_input_offset(self.nodes[i].edges_out[j].get_input(), offset);
                }
                offset += output_len as usize;
            }
        }*/
        self.tape_size = offset;
        
        // we  need to figure out the order
        // first find nodes with no inputs
        /*let mut output_list: Vec<BlockPtr> = Vec::new();
        let mut ready: Vec<BlockPtr> = Vec::new(); // we will use this as a list of ready nodes to choose from
        let mut num_inputs_done: Vec<usize> = Vec::new();
        for i in 0..self.len() {
            if self.nodes[i].edges_in.len() == 0 {ready.push(BlockPtr(i))};
            num_inputs_done.push(0);
        }
        
        
        let mut num_steps: u32 = 0;
        let mut new_ready : Vec<BlockPtr> = Vec::new();
            
        while (ready.len() > 0) && (num_steps < 1000) {
            num_steps += 1;
            new_ready.truncate(0);
            for bp in ready.iter() {
                let current_node_id = bp.get_node_id();
                output_list.push(*bp);
                for x in self.nodes[bp.get_node_id()].edges_out.iter() {
                    let out_node_id = x.get_node_id();
                    num_inputs_done[out_node_id] += 1;
                    if self.nodes[out_node_id].edges_in.len() == num_inputs_done[out_node_id] {
                        new_ready.push(BlockPtr(out_node_id));
                    }
                }
            }
            swap(&mut new_ready, &mut ready)
            
        }
        
        println!("Outputs: {:?}", output_list);
        if ready.len() > 0 && num_steps == 1000 {
            panic!("Couldn't resolve the graph in 1000 steps");
        }
         
        */
                
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
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);  
    }
    
    #[test]
    fn graph_one_sink() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in.len(), 0);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);  

        // Let's add one result block 
        let mut output_node = block_misc::new_result_block(&mut bg, const_block_output, 1.0).unwrap();
        assert_eq!(output_node, ());
//        println!("Output nodes: {:?}", output_nodes);
        //println!("Block 0: edges_in: {:?}, edges_out: {:?}", bg.edges_in[0], bg.edges_out[0]);
        assert_eq!(bg.nodes[0].edges_in, vec![]);      // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]);  

        assert_eq!(bg.nodes[1].edges_in, vec![BlockPtrOutput(BlockPtr(0), OutputSlot(0))]);
        assert_eq!(bg.nodes[1].edges_out, vec![]);  
    }
    
    #[test]
    fn graph_two_sinks() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output, BlockPtrOutput(BlockPtr(0), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // We need to add a copy block to have two sinks
        let mut copies = block_misc::new_copy_block2(&mut bg, const_block_output).unwrap();
        let c2 = copies.pop().unwrap();
        let c1 = copies.pop().unwrap();
        
        
        // Let's add one result block 
        let mut output_node = block_misc::new_result_block(&mut bg, c1, 1.0).unwrap();
        assert_eq!(output_node, ());
//        println!("Output nodes: {:?}", output_nodes);
        //println!("Block 0: edges_in: {:?}, edges_out: {:?}", bg.edges_in[0], bg.edges_out[0]);
        assert_eq!(bg.nodes[0].edges_in.len(), 0);      
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]);	  	

        // Let's add second result block to see what happens

        let mut output_node = block_misc::new_result_block(&mut bg, c2, 1.0).unwrap();
        assert_eq!(output_node, ());
        assert_eq!(bg.nodes[0].edges_in, vec![]);      
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]);
        // copy block
        assert_eq!(bg.nodes[1].edges_in, vec![BlockPtrOutput(BlockPtr(0), OutputSlot(0))]);
        assert_eq!(bg.nodes[1].edges_out, vec![BlockPtrInput(BlockPtr(2), InputSlot(0)), BlockPtrInput(BlockPtr(3), InputSlot(0))]);
        // result block 1
        assert_eq!(bg.nodes[2].edges_in, vec![BlockPtrOutput(BlockPtr(1), OutputSlot(0))]);
        assert_eq!(bg.nodes[2].edges_out, vec![]);
        // result bock 2
        assert_eq!(bg.nodes[3].edges_in, vec![BlockPtrOutput(BlockPtr(1), OutputSlot(1))]);
        assert_eq!(bg.nodes[3].edges_out, vec![]);
    }
    
    
    
    #[test]
    fn graph_two_sources() {
        let mut bg = BlockGraph::new();
        
        let const_block_output1 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(const_block_output1, BlockPtrOutput(BlockPtr(0), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        let const_block_output2 = block_misc::new_const_block(&mut bg, vec![1.0, 2.0]).unwrap();
        assert_eq!(const_block_output2, BlockPtrOutput(BlockPtr(1), OutputSlot(0)));
        assert_eq!(bg.nodes[1].edges_in, vec![]);
        assert_eq!(bg.nodes[1].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // Using the join block, we merge two outputs into one single output (copy-less implementation)
        let mut union_output = block_misc::new_join_block2(&mut bg, vec![const_block_output1, const_block_output2]).unwrap();
        assert_eq!(union_output, BlockPtrOutput(BlockPtr(2), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BlockPtrInput(BlockPtr(2), InputSlot(0))]);
        assert_eq!(bg.nodes[1].edges_in, vec![]);
        assert_eq!(bg.nodes[1].edges_out, vec![BlockPtrInput(BlockPtr(2), InputSlot(1))]);
        
        // the join block 
        assert_eq!(bg.nodes[2].edges_in, vec![BlockPtrOutput(BlockPtr(0), OutputSlot(0)), BlockPtrOutput(BlockPtr(1), OutputSlot(0))]);
        assert_eq!(bg.nodes[2].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
    }    


    #[test]
    fn schedule_simple() {
        let mut bg = BlockGraph::new();
        
        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let output_node = block_misc::new_result_block(&mut bg, const_block_output, 1.0).unwrap();
        assert_eq!(output_node, ());
        let list = bg.schedule();
        
    }

    
    
}
