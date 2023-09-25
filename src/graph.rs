#![allow(dead_code,unused_imports)]

use crate::block_misc;
use crate::model_instance;
use crate::port_buffer;
use crate::regressor::BlockTrait;
use std::error::Error;
use std::mem;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OutputSlot(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InputSlot(usize);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtr(usize); // just an id in a graph
#[derive(Debug, PartialEq)]
pub struct BlockPtrOutput(BlockPtr, OutputSlot); // since blocks can have multiple outputs, separate between them
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockPtrInput(BlockPtr, InputSlot); // since blocks can have multiple inputs, separate between them

#[derive(Debug)]
pub struct BlockGraphNode {
    pub edges_in: Vec<BlockPtrOutput>, // each block can have multiple input edges
    pub edges_out: Vec<BlockPtrInput>, // each block can have multiple output edges
}

pub struct BlockGraph {
    nodes: Vec<BlockGraphNode>,
    blocks: Vec<Box<dyn BlockTrait>>,
    pub blocks_final: Vec<Box<dyn BlockTrait>>,
    tape_size: usize,
}

// We need to treat join type in a special way - all inputs need to be consequtive
#[derive(PartialEq, Debug)]
pub enum BlockType {
    Regular,
    Join,
    Observe,
    Copy,
}

impl BlockPtr {
    pub fn get_node_id(&self) -> usize {
        self.0
    }
}

impl OutputSlot {
    pub fn get_output_index(&self) -> usize {
        self.0
    }
}

impl InputSlot {
    pub fn get_input_index(&self) -> usize {
        self.0
    }
}

impl BlockPtrOutput {
    pub fn get_block_ptr(&self) -> BlockPtr {
        self.0
    }
    pub fn get_node_id(&self) -> usize {
        self.0.get_node_id()
    }
    pub fn get_output_index(&self) -> usize {
        self.1.get_output_index()
    }
    pub fn get_output(&self) -> OutputSlot {
        self.1
    }
}

impl BlockPtrInput {
    pub fn get_block_ptr(&self) -> BlockPtr {
        self.0
    }
    pub fn get_node_id(&self) -> usize {
        self.0.get_node_id()
    }
    pub fn get_input_index(&self) -> usize {
        self.1.get_input_index()
    }
    pub fn get_input(&self) -> InputSlot {
        self.1
    }
}

const BLOCK_PTR_INPUT_DEFAULT: BlockPtrInput =
    BlockPtrInput(BlockPtr(usize::MAX), InputSlot(usize::MAX));

impl BlockGraph {
    pub fn new() -> BlockGraph {
        BlockGraph {
            nodes: Vec::new(),
            blocks: Vec::new(),
            blocks_final: Vec::new(),
            tape_size: usize::MAX,
        }
    }

    pub fn add_node(
        &mut self,
        block: Box<dyn BlockTrait>,
        edges_in: Vec<BlockPtrOutput>,
    ) -> Result<Vec<BlockPtrOutput>, Box<dyn Error>> {
        // Due to how CopyBlock works (zero-copy first ouptut), it's first output cannot go to a Join block since join block needs to control its inputs
        // TODO we could just insert TrueCopy block here...

        let mut edges_in = edges_in;
        if block.get_block_type() == BlockType::Join {
            // Join a is a special block, because it does zero copy joining of outputs
            let mut new_edges_in: Vec<BlockPtrOutput> = Vec::new();
            for edge_in in edges_in.into_iter() {
                let node_id_in = edge_in.get_node_id();
                if self.blocks[node_id_in].get_block_type() == BlockType::Join {
                    // Join -> Join can always be merged into a single join
                    // So we won't add a block here, instead, we will add input edges of the previous block
                    // And abandon the previous block with empty inputs and outputs.
                    new_edges_in.append(&mut self.nodes[node_id_in].edges_in);
                    self.nodes[node_id_in].edges_out.truncate(0);
                } else {
                    new_edges_in.push(edge_in);
                }
            }
            edges_in = new_edges_in;
        } else if block.get_block_type() == BlockType::Copy {
            // if we connect copy to copy... we simply increase number of copies of existing input block
            assert_eq!(edges_in.len(), 1);
            let edge_in = &edges_in[0];
            let node_id_in = edge_in.get_node_id();
            if self.blocks[node_id_in].get_block_type() == BlockType::Copy {
                let copy_block = self.blocks[node_id_in]
                    .as_any()
                    .downcast_mut::<block_misc::BlockCopy>()
                    .unwrap();
                let bp = BlockPtr(node_id_in);
                let bo = BlockPtrOutput(bp, OutputSlot(copy_block.output_offsets.len()));
                copy_block.output_offsets.push(usize::MAX);
                self.nodes[node_id_in]
                    .edges_out
                    .push(BLOCK_PTR_INPUT_DEFAULT); // make empty spaceg
                return Ok(vec![bo, edges_in.pop().unwrap()]);
            }
        }

        let num_output_connectors = block.get_num_output_slots();
        let bp = BlockPtr(self.nodes.len()); // id of current node
        for (i, e) in edges_in.iter().enumerate() {
            let bi = InputSlot(i);
            let bpi = BlockPtrInput(bp, bi);
            self.nodes[e.get_node_id()].edges_out[e.get_output_index()] = bpi;
        }
        let newnode = BlockGraphNode {
            edges_in,
            edges_out: Vec::new(),
        };

        self.nodes.push(newnode);
        self.blocks.push(block);
        let mut vo: Vec<BlockPtrOutput> = Vec::new();
        for i in 0..num_output_connectors {
            let bo = BlockPtrOutput(bp, OutputSlot(i));
            vo.push(bo);
            self.nodes[bp.get_node_id()]
                .edges_out
                .push(BLOCK_PTR_INPUT_DEFAULT); // make empty spaceg
        }
        Ok(vo)
    }

    pub fn println(&self) {
        log::info!("Graph nodes:\n");
        for n in self.nodes.iter() {
            println!("  {:?}", n);
        }
    }

    pub fn get_tape_size(&self) -> usize {
        assert_ne!(
            self.tape_size,
            usize::MAX,
            "get_tape_size() called on a graph before calling finalize()"
        );
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
        assert!(
            !self.blocks_final.is_empty(),
            "There are no blocks in the final graph? Have you called finalize() yet?"
        );
        for i in 0..self.blocks_final.len() {
            self.blocks_final[i].allocate_and_init_weights(mi);
        }
    }

    pub fn take_blocks(&mut self) -> Vec<Box<dyn BlockTrait>> {
        mem::take(&mut self.blocks_final)

        /*
        let mut blocks : Vec<Box<dyn BlockTrait>> = Vec::new();
        for block in mem::take(&mut self.blocks).into_iter() {
            // Join Block is a no-op, so it doesn't need to be executed. This is a super small optimization.
            if block.get_block_type() != BlockType::Join {
                blocks.push(block);
            }
        }
        blocks */
    }

    pub fn finalize(&mut self) {
        let mut offset: usize = 0;

        // Let's first install sinks, so the graph is without dangling parts
        let mut sinks: Vec<BlockPtrOutput> = Vec::new();
        for i in 0..self.len() {
            for (output_index, edge_out) in self.nodes[i].edges_out.iter().enumerate() {
                if *edge_out == BLOCK_PTR_INPUT_DEFAULT {
                    let bptro = BlockPtrOutput(BlockPtr(i), OutputSlot(output_index));
                    sinks.push(bptro);
                }
            }
        }
        // TODO we could have a single sink for all
        for bptro in sinks.into_iter() {
            // For neural nets, zeroing out the backward data is the least-surprise way of doing it
            block_misc::new_sink_block(self, bptro, block_misc::SinkType::Zero).unwrap();
        }

        // Now allocate inputs/outputs to parts of the tape
        for i in 0..self.len() {
            let current_block_type = self.blocks[i].get_block_type();

            for (input_index, edge_in) in self.nodes[i].edges_in.iter().enumerate() {
                let bo = edge_in.get_output();
                let bptr = edge_in.get_node_id();
                let output_len = self.blocks[bptr].get_num_output_values(bo);
                let input_block_type = self.blocks[bptr].get_block_type();
                if (input_block_type == BlockType::Join)
                    || (input_block_type == BlockType::Observe)
                    || (input_block_type == BlockType::Copy)
                        && (bo.get_output_index() == 0
                            && current_block_type != BlockType::Join
                            && current_block_type != BlockType::Observe)
                {
                    // we are special casing Join block
                    // It is zero-copy joining of inputs, which means inputs and outputs share exactly the same space
                    let fake_offset = self.blocks[bptr].get_input_offset(InputSlot(0)).unwrap();
                    self.blocks[bptr].set_output_offset(bo, fake_offset);
                    self.blocks[i].set_input_offset(InputSlot(input_index), fake_offset);
                } else if (input_block_type == BlockType::Regular)
                    || (input_block_type == BlockType::Copy)
                {
                    self.blocks[bptr].set_output_offset(bo, offset);
                    self.blocks[i].set_input_offset(InputSlot(input_index), offset);
                    offset += output_len;
                } else {
                    panic!(
                        "Type of block not supported in scheduling: {:?}",
                        input_block_type
                    );
                }
            }
        }
        self.tape_size = offset;

        // Prepare the final list of blocks
        for block in mem::take(&mut self.blocks).into_iter() {
            // Join Block is a no-op, so it doesn't need to be executed. This is a super small optimization.
            if block.get_block_type() != BlockType::Join {
                self.blocks_final.push(block);
            }
        }
    }
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_ffm;
    use crate::block_loss_functions;
    use crate::block_lr;
    use crate::block_misc;
    use crate::block_misc::Observe;
    use crate::model_instance;
    use crate::model_instance::Optimizer;

    #[test]
    fn graph_creation() {
        let mut bg = BlockGraph::new();

        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(
            const_block_output,
            BlockPtrOutput(BlockPtr(0), OutputSlot(0))
        );
        assert_eq!(bg.nodes[0].edges_in, vec![]); // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
    }

    #[test]
    fn graph_one_sink() {
        let mut bg = BlockGraph::new();

        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(
            const_block_output,
            BlockPtrOutput(BlockPtr(0), OutputSlot(0))
        );
        assert_eq!(bg.nodes[0].edges_in.len(), 0); // basically []
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // Let's add one result block
        let _output_node =
            block_misc::new_observe_block(&mut bg, const_block_output, Observe::Forward, Some(1.0))
                .unwrap();

        assert_eq!(bg.nodes[0].edges_in, vec![]); // basically []
        assert_eq!(
            bg.nodes[0].edges_out,
            vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]
        );

        assert_eq!(
            bg.nodes[1].edges_in,
            vec![BlockPtrOutput(BlockPtr(0), OutputSlot(0))]
        );
        assert_eq!(bg.nodes[1].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
    }

    #[test]
    fn graph_two_sinks() {
        let mut bg = BlockGraph::new();

        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(
            const_block_output,
            BlockPtrOutput(BlockPtr(0), OutputSlot(0))
        );
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // We need to add a copy block to have two sinks
        let (c1, c2) = block_misc::new_copy_block_2(&mut bg, const_block_output).unwrap();

        // Let's add one result block
        let _output_node =
            block_misc::new_observe_block(&mut bg, c1, Observe::Forward, Some(1.0)).unwrap();

        assert_eq!(bg.nodes[0].edges_in.len(), 0);
        assert_eq!(
            bg.nodes[0].edges_out,
            vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]
        );

        // Let's add second result block to see what happens

        let _output_node =
            block_misc::new_observe_block(&mut bg, c2, Observe::Forward, Some(1.0)).unwrap();
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(
            bg.nodes[0].edges_out,
            vec![BlockPtrInput(BlockPtr(1), InputSlot(0))]
        );
        // copy block
        assert_eq!(
            bg.nodes[1].edges_in,
            vec![BlockPtrOutput(BlockPtr(0), OutputSlot(0))]
        );
        assert_eq!(
            bg.nodes[1].edges_out,
            vec![
                BlockPtrInput(BlockPtr(2), InputSlot(0)),
                BlockPtrInput(BlockPtr(3), InputSlot(0))
            ]
        );
        // result block 1
        assert_eq!(
            bg.nodes[2].edges_in,
            vec![BlockPtrOutput(BlockPtr(1), OutputSlot(0))]
        );
        assert_eq!(bg.nodes[2].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
        // result bock 2
        assert_eq!(
            bg.nodes[3].edges_in,
            vec![BlockPtrOutput(BlockPtr(1), OutputSlot(1))]
        );
        assert_eq!(bg.nodes[3].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
    }

    #[test]
    fn graph_two_sources() {
        let mut bg = BlockGraph::new();

        let const_block_output1 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        assert_eq!(
            const_block_output1,
            BlockPtrOutput(BlockPtr(0), OutputSlot(0))
        );
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(bg.nodes[0].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        let const_block_output2 = block_misc::new_const_block(&mut bg, vec![1.0, 2.0]).unwrap();
        assert_eq!(
            const_block_output2,
            BlockPtrOutput(BlockPtr(1), OutputSlot(0))
        );
        assert_eq!(bg.nodes[1].edges_in, vec![]);
        assert_eq!(bg.nodes[1].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);

        // Using the join block, we merge two outputs into one single output (copy-less implementation)
        let union_output =
            block_misc::new_join_block(&mut bg, vec![const_block_output1, const_block_output2])
                .unwrap();
        assert_eq!(union_output, BlockPtrOutput(BlockPtr(2), OutputSlot(0)));
        assert_eq!(bg.nodes[0].edges_in, vec![]);
        assert_eq!(
            bg.nodes[0].edges_out,
            vec![BlockPtrInput(BlockPtr(2), InputSlot(0))]
        );
        assert_eq!(bg.nodes[1].edges_in, vec![]);
        assert_eq!(
            bg.nodes[1].edges_out,
            vec![BlockPtrInput(BlockPtr(2), InputSlot(1))]
        );

        // the join block
        assert_eq!(
            bg.nodes[2].edges_in,
            vec![
                BlockPtrOutput(BlockPtr(0), OutputSlot(0)),
                BlockPtrOutput(BlockPtr(1), OutputSlot(0))
            ]
        );
        assert_eq!(bg.nodes[2].edges_out, vec![BLOCK_PTR_INPUT_DEFAULT]);
    }

    #[test]
    fn finalize_simple() {
        let mut bg = BlockGraph::new();

        let const_block_output = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let _output_node =
            block_misc::new_observe_block(&mut bg, const_block_output, Observe::Forward, Some(1.0))
                .unwrap();
        bg.finalize();
        assert_eq!(bg.tape_size, 1);
    }

    #[test]
    fn finalize_realistic() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();

        let re_lr = block_lr::new_lr_block(&mut bg, &mi).unwrap();
        let re_ffm = block_ffm::new_ffm_block(&mut bg, &mi).unwrap();
        let joined = block_misc::new_join_block(&mut bg, vec![re_lr, re_ffm]).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, joined, true);
        bg.finalize();
    }

    #[test]
    fn finalize_copy_to_join() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.ffm_k = 1;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used
        mi.optimizer = Optimizer::AdagradLUT;

        let mut bg = BlockGraph::new();

        let const_1 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap(); // 0
        let const_2 = block_misc::new_const_block(&mut bg, vec![3.0]).unwrap(); // 1
        let const_3 = block_misc::new_const_block(&mut bg, vec![4.0]).unwrap(); // 2
        let const_4 = block_misc::new_const_block(&mut bg, vec![4.0]).unwrap(); // 3
        let (copy_output_1, copy_output_2) =
            block_misc::new_copy_block_2(&mut bg, const_1).unwrap(); // 4
        let (_copy_output_3, _copy_output_4) =
            block_misc::new_copy_block_2(&mut bg, const_2).unwrap(); // 5

        // this is not zero copy
        let _join_1 = block_misc::new_join_block(&mut bg, vec![copy_output_1, const_3]).unwrap(); // 6
                                                                                                  // this is zero copy
        let _join_2 = block_misc::new_join_block(&mut bg, vec![const_4, copy_output_2]); // 7
        bg.finalize();
        let mut list = bg.take_blocks();

        {
            // first one goes to copy again, so it cannot use input as an output
            let copy_block_1 = list[4]
                .as_any()
                .downcast_mut::<block_misc::BlockCopy>()
                .unwrap();
            assert_ne!(copy_block_1.input_offset, copy_block_1.output_offsets[0]);
        }
        {
            // But second one can re-use the input as output
            let copy_block_2 = list[5]
                .as_any()
                .downcast_mut::<block_misc::BlockCopy>()
                .unwrap();
            assert_eq!(copy_block_2.input_offset, copy_block_2.output_offsets[0]);
        }
    }

    #[test]
    fn finalize_join_to_join() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.ffm_k = 1;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used
        mi.optimizer = Optimizer::AdagradLUT;

        let mut bg = BlockGraph::new();

        let const_1 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let const_2 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let const_3 = block_misc::new_const_block(&mut bg, vec![1.0]).unwrap();
        let join_block1 = block_misc::new_join_block(&mut bg, vec![const_1, const_2]).unwrap();
        let _join_block2 = block_misc::new_join_block(&mut bg, vec![join_block1, const_3]).unwrap();
        assert_eq!(bg.nodes.len(), 5);
        assert_eq!(bg.nodes[3].edges_in.len(), 0); // 3 is the first join block which was removed
        assert_eq!(bg.nodes[3].edges_out.len(), 0);
        assert_eq!(bg.nodes[4].edges_in.len(), 3); // now fourth block has 3 inputs, not 2

        bg.finalize();
        let list = bg.take_blocks();
        assert_eq!(list.len(), 4); // both join blocks are no-op and thus not returned, but sink block is added automatically
    }
}
