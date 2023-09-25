#![allow(dead_code,unused_imports)]

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
            hash_seed,
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

    pub(crate) fn insert(&mut self, key: &[u8], value: NamespaceDescriptorWithHash) {
        let mut node = &mut self.root;

        for &byte in key {
            let child = &mut node.children[byte as usize];
            node = child.get_or_insert_with(RadixTreeNode::new);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vwmap::{NamespaceFormat, NamespaceType};

    #[test]
    fn test_insert_and_get() {
        let mut tree = RadixTree::new();

        let namespace_descriptor_with_hash_1 = NamespaceDescriptorWithHash {
            descriptor: NamespaceDescriptor {
                namespace_index: 0,
                namespace_type: NamespaceType::Primitive,
                namespace_format: NamespaceFormat::Categorical,
            },
            hash_seed: 1,
        };

        let namespace_descriptor_with_hash_2 = NamespaceDescriptorWithHash {
            descriptor: NamespaceDescriptor {
                namespace_index: 10,
                namespace_type: NamespaceType::Primitive,
                namespace_format: NamespaceFormat::Categorical,
            },
            hash_seed: 2,
        };

        let namespace_descriptor_with_hash_3 = NamespaceDescriptorWithHash {
            descriptor: NamespaceDescriptor {
                namespace_index: 20,
                namespace_type: NamespaceType::Primitive,
                namespace_format: NamespaceFormat::Categorical,
            },
            hash_seed: 3,
        };

        tree.insert(b"A", namespace_descriptor_with_hash_1);
        tree.insert(b"AB", namespace_descriptor_with_hash_2);
        tree.insert(b"ABC", namespace_descriptor_with_hash_3);

        assert_eq!(tree.get(b"A"), Some(&namespace_descriptor_with_hash_1));
        assert_eq!(tree.get(b"AB"), Some(&namespace_descriptor_with_hash_2));
        assert_eq!(tree.get(b"ABC"), Some(&namespace_descriptor_with_hash_3));
        assert_eq!(tree.get(b"ABCD"), None);
    }

    #[test]
    fn test_insert_and_get_empty_key() {
        let mut tree = RadixTree::new();

        let namespace_descriptor_with_hash = NamespaceDescriptorWithHash {
            descriptor: NamespaceDescriptor {
                namespace_index: 0,
                namespace_type: NamespaceType::Primitive,
                namespace_format: NamespaceFormat::Categorical,
            },
            hash_seed: 1,
        };

        tree.insert(b"", namespace_descriptor_with_hash);

        assert_eq!(tree.get(b""), Some(&namespace_descriptor_with_hash));
        assert_eq!(tree.get(b"A"), None);
    }

    #[test]
    fn test_insert_and_get_long_key() {
        let mut tree = RadixTree::new();

        let namespace_descriptor_with_hash = NamespaceDescriptorWithHash {
            descriptor: NamespaceDescriptor {
                namespace_index: 0,
                namespace_type: NamespaceType::Primitive,
                namespace_format: NamespaceFormat::Categorical,
            },
            hash_seed: 1,
        };

        tree.insert(b"AB", namespace_descriptor_with_hash);

        assert_eq!(tree.get(b"AB"), Some(&namespace_descriptor_with_hash));
        assert_eq!(tree.get(b"A"), None);
        assert_eq!(tree.get(b"B"), None);
        assert_eq!(tree.get(b"ABC"), None);
    }
}
