use crate::regressor::Regressor;
use core::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::mem;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::Mutex;

// This is a helper for UNSAFELY sharing data between threads

pub struct UnsafelySharableTrait<T: Sized> {
    content: ManuallyDrop<T>,
    reference_count: Arc<Mutex<PhantomData<u32>>>,
}

pub type BoxedRegressorTrait = UnsafelySharableTrait<Box<Regressor>>;

// SUPER UNSAFE
// SUPER UNSAFE
// SUPER UNSAFE
// This literary means we are on our own -- but it is the only way to implement HogWild performantly
unsafe impl<T: Sized> Sync for UnsafelySharableTrait<T> {}
unsafe impl<T: Sized> Send for UnsafelySharableTrait<T> {}

impl<T: Sized> Deref for UnsafelySharableTrait<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.content
    }
}

impl<T: Sized> DerefMut for UnsafelySharableTrait<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.content
    }
}

impl<T: Sized> Drop for UnsafelySharableTrait<T> {
    fn drop(&mut self) {
        unsafe {
            // we are called before reference is removed, so we need to decide if to drop it or not
            let count = Arc::<Mutex<PhantomData<u32>>>::strong_count(&self.reference_count) - 1;
            if count == 0 {
                let _box_to_be_dropped = ManuallyDrop::take(&mut self.content);
                // Now this means that the content will be dropped
            }
        }
    }
}

impl<T: Sized + 'static> UnsafelySharableTrait<T> {
    pub fn new(a: T) -> UnsafelySharableTrait<T> {
        UnsafelySharableTrait::<T> {
            content: ManuallyDrop::new(a),
            reference_count: Arc::new(Mutex::new(std::marker::PhantomData {})),
        }
    }
}

// Non-generalized implementation
// Todo - generalize this[A

impl BoxedRegressorTrait {
    pub fn clone(&self) -> BoxedRegressorTrait {
        // UNSAFE AS HELL
        unsafe {
            // Double deref here sounds weird, but you got to know that dyn Trait and Box<dyn Trait> are the same thing, just box owns it.
            // And you can get dyn Trait content, but you can't get box content (directly)
            let r2: Box<Regressor> = mem::transmute(self.content.deref().deref());

            BoxedRegressorTrait {
                content: ManuallyDrop::new(r2),
                reference_count: self.reference_count.clone(),
            }
        }
    }
}
