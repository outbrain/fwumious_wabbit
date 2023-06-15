use crate::vwmap::NamespaceDescriptor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NamespaceDescriptorWithHash {
    pub(crate) descriptor: NamespaceDescriptor,
    pub(crate) hash_seed: u32,
}

impl NamespaceDescriptorWithHash {
    pub(crate) fn new(descriptor: NamespaceDescriptor, hash_seed: u32) -> Self {
        Self {
            descriptor,
            hash_seed
        }
    }
}

#[derive(Clone, Debug)]
struct RadixTreeNode {
    children: Vec<Option<RadixTreeNode>>,
    value: Option<NamespaceDescriptorWithHash>,
}

impl Default for RadixTreeNode {
    fn default() -> Self {
        Self {
            children: vec![None; 256],
            value: None,
        }
    }
}

impl RadixTreeNode {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Default, Debug)]
pub struct RadixTree {
    root: RadixTreeNode,
}

impl RadixTree {
    fn new() -> Self {
        RadixTree {
            root: RadixTreeNode::new(),
        }
    }

    pub(crate)fn insert(&mut self, key: &[u8], value: NamespaceDescriptorWithHash) {
        let mut node = &mut self.root;

        for &byte in key {
            let child = &mut node.children[byte as usize];
            node = child.get_or_insert_with(|| RadixTreeNode::new());
        }

        node.value = Some(value);
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<&NamespaceDescriptorWithHash> {
        let mut node = &self.root;

        for &byte in key {
            let maybe_child = &node.children[byte as usize];
            if let Some(child) = maybe_child {
                node = child;
            } else {
                return None.as_ref();
            }
        }

        node.value.as_ref()
    }
}